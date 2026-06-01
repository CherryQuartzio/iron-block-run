"""
Iron Block Run — MineRL Horse Racing PPO Agent
================================================
Trains a reinforcement learning agent to complete a race track on horseback
in Minecraft 1.16.5 using Proximal Policy Optimization (PPO).

Requires:
    - MineRL v1.0.2 (pip install git+https://github.com/minerllabs/minerl)
    - stable-baselines3
    - matplotlib
    - A valid Minecraft 1.16.5 world folder at WORLD_DIR (see config below)

Usage:
    xvfb-run python agent.py
"""
import os
import logging
import collections
import zipfile
import tempfile
import math
import shutil
from typing import List

import nbtlib
import gym
from typing import Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
from PIL import Image
import minerl  # noqa: F401 — registers built-in MineRL envs on import
# Optional OpenCV for visualization. It's safe if not installed; visualization
# remains disabled by default.
try:
    import cv2
    print("USING CV2")
    _CV2_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    cv2 = None
    print("NOT USING CV2")
    _CV2_AVAILABLE = False
VISUALIZE = True
from minerl.herobraine.env_spec import EnvSpec
from minerl.herobraine.env_specs.human_controls import HumanControlEnvSpec
from minerl.herobraine.hero import handlers
from minerl.herobraine.hero.handlers.agent.start import (
    AgentStartPlacement,
    DoneOnDeath,
    LoadWorldAgentStart,
)
from minerl.herobraine.hero.handler import Handler
from minerl.herobraine.hero.handlers.translation import TranslationHandler

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import BaseCallback

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------

# -- Paths --
WORLD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "world")
WORLD_ZIP = None  # Set at runtime by prepare_world_zip()

# -- Agent spawn --
SPAWN_X = -73.0
SPAWN_Y = 71.0      # Ground level at spawn (world block surface ~Y=70)
SPAWN_Z = -149.0
SPAWN_YAW = 180.0    # Facing -Z (toward the race track start)

# -- Horse spawn (directly in front of agent) --
# Blocks ahead of the agent that the DrawEntity decorator spawns the horse.
HORSE_DISTANCE = 3
# Max env steps (ticks) spent walking up to and mounting the horse. Each tick
# is ~0.22 blocks of walking, so this must comfortably exceed HORSE_DISTANCE.
MOUNT_MAX_STEPS = 25

# Degrees to pitch the camera down before mounting. At pitch 0 the interaction
# ray points at the horizon and sails over the horse; looking down aims it at
# the horse's body so 'use' actually targets (and mounts) the horse.
MOUNT_LOOK_DOWN_DEG = 20.0
_HORSE_YAW_RAD = math.radians(SPAWN_YAW)
HORSE_X = SPAWN_X - math.sin(_HORSE_YAW_RAD) * HORSE_DISTANCE
HORSE_Y = SPAWN_Y       # Same ground level as agent
HORSE_Z = SPAWN_Z + math.cos(_HORSE_YAW_RAD) * HORSE_DISTANCE

# Applied in patches/EnvServer.java configureSpawnedHorse() (not via DrawEntity NBT).
HORSE_MOVEMENT_SPEED = 0.2
HORSE_JUMP_STRENGTH = 0.85
HORSE_MAX_HEALTH = 20.0
HORSE_HEALTH = 20.0
HORSE_VARIANT = 1029

# -- Observation --
OBS_WIDTH = 144
OBS_HEIGHT = 144
NATIVE_RES = (640, 360)  # MineRL v1.0.2 native POV resolution (width, height)

# -- Training hyperparameters --
TOTAL_TIMESTEPS = 100_000
LEARNING_RATE = 3e-4
N_STEPS = 2048       # Steps per rollout buffer collection
BATCH_SIZE = 64
N_EPOCHS = 10        # PPO epochs per update
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
MAX_EPISODE_STEPS = 6000  # ~5 minutes at 20 tps

# -- Output --
MODEL_SAVE_PATH = "horse_race_ppo"
REWARD_PLOT_PATH = "training_rewards.png"
RACE_TIME_PLOT_PATH = "training_race_time.png"
SEGMENT_TIME_PLOT_PATH = "training_segment_times.png"
COMPLETION_RATE_PLOT_PATH = "training_completion_rate.png"

# -- Timing --
TICKS_PER_SECOND = 20  # Minecraft runs at 20 ticks per second

# -- Reward constants (tunable) --
REWARD_CHECKPOINT = 50.0        # Crossing the next expected checkpoint
REWARD_LAP_COMPLETE = 200.0     # Crossing start/goal after all checkpoints
REWARD_PROGRESS = 0.1           # Multiplier for distance-decrease toward next CP
REWARD_ON_PATH = 0.05           # Per-step: horse on grass_path / dirt_path
REWARD_GOLD_BLOCK = 2.0         # Stepped on gold_block (speed boost plate)
REWARD_SPRUCE_SLAB = 1.0        # Standing on spruce_slab (bridge over water)
PENALTY_SOUL_SAND = -0.5        # Per-step: on soul_sand
PENALTY_WATER = -0.5            # Per-step: in water
PENALTY_COBWEB = -1.0           # Per-step: in cobweb
PENALTY_OFF_COURSE = -0.3       # Per-step: on grass_block (off track)
PENALTY_TIME = -0.01            # Per-step: encourages speed
PENALTY_STUCK = -5.0            # Terminal: stuck too long
PENALTY_FAR_OFF_COURSE = -5.0   # Terminal: too far from track
PENALTY_WRONG_DIRECTION = -1.0  # Per-step: moving toward previous CP (backward)

# -- Stuck / off-course detection --
STUCK_WINDOW = 100              # Steps to check for stuck condition
STUCK_MIN_DISPLACEMENT = 1.0    # Minimum blocks moved in STUCK_WINDOW steps
OFF_COURSE_MAX_DIST = 30.0      # Max distance from nearest track segment

# -- Horse height offset --
# Player riding a horse has ypos ~1.6-2.0 blocks above the ground block.
# When querying blocks below, we account for this offset.
HORSE_Y_OFFSET = 2  # Grid y-offset to reach ground level from player pos

# -- Checkpoints --
# Each checkpoint is ((x1,z1), (x2,z2)). Axis indicates which coord is
# constant (the "gate" axis), and the agent crosses by changing that coord.
# fmt: off
CHECKPOINTS = [
    # Start/Goal line: X varies -70..-78, Z fixed at -165
    {"name": "Start",  "p1": (-70, -165), "p2": (-78, -165), "axis": "z"},
    # Checkpoint A: X varies -78..-69, Z fixed at -217
    {"name": "CP_A",   "p1": (-78, -217), "p2": (-69, -217), "axis": "z"},
    # Checkpoint B: X varies -30..-20, Z fixed at -305
    {"name": "CP_B",   "p1": (-30, -305), "p2": (-20, -305), "axis": "z"},
    # Checkpoint C: Z varies -376..-385, X fixed at -39
    {"name": "CP_C",   "p1": (-39, -376), "p2": (-39, -385), "axis": "x"},
    # Checkpoint D: Z varies -363..-376, X fixed at -100
    {"name": "CP_D",   "p1": (-100, -363), "p2": (-100, -376), "axis": "x"},
    # Checkpoint E: X varies -110..-120, Z fixed at -268
    {"name": "CP_E",   "p1": (-110, -268), "p2": (-120, -268), "axis": "z"},
]
# fmt: on
NUM_CHECKPOINTS = len(CHECKPOINTS)  # 6 (including start/goal)

# -- Block type vocabulary for grid observation --
BLOCK_TO_ID = {
    "air": 0,
    "grass_path": 1, "dirt_path": 1,  # dirt_path is 1.17+ name for grass_path
    "grass_block": 2,
    "soul_sand": 3,
    "water": 4, "flowing_water": 4,
    "cobweb": 5,
    "gold_block": 6,
    "spruce_slab": 7,
    "light_weighted_pressure_plate": 8,
    "oak_fence": 9,
    "stone": 10, "cobblestone": 10, "stone_bricks": 10,
    "dirt": 11,
}
BLOCK_UNKNOWN_ID = 99

# -- Charge jump macro --
CHARGE_JUMP_TICKS = 10  # How many ticks to hold jump for fence clearing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===========================================================================
#  0. World Zip Preparation
# ===========================================================================

def _sanitize_world_staging(staging_dir: str) -> None:
    """Force survival mode and drop saved player state from the staging copy."""
    level_dat = os.path.join(staging_dir, "level.dat")
    if os.path.isfile(level_dat):
        nbt = nbtlib.load(level_dat)
        if "Data" in nbt:
            nbt["Data"]["GameType"] = nbtlib.Int(0)
            nbt["Data"]["allowCommands"] = nbtlib.Byte(0)
        nbt.save(level_dat)

    playerdata_dir = os.path.join(staging_dir, "playerdata")
    if os.path.isdir(playerdata_dir):
        for fname in os.listdir(playerdata_dir):
            if fname.endswith(".dat"):
                os.remove(os.path.join(playerdata_dir, fname))


def prepare_world_zip(world_dir: str) -> str:
    """
    Create a zip file of the custom world in the structure expected by
    MineRL's Java ``ReplaySender.loadWorldFromZip()``.

    The Java code expects the zip to contain files under:
        ``<prefix>/saves/<world_name>/level.dat``
        ``<prefix>/saves/<world_name>/region/...``

    It extracts the world name as ``entries[0].split("/")[2]`` (third
    path component of the first zip entry), then calls
    ``Minecraft.loadWorld(extractDir/saves, worldName)``.

    Returns:
        The absolute path to the generated zip file.
    """
    global WORLD_ZIP

    if not os.path.isdir(world_dir):
        raise FileNotFoundError(
            f"Custom world directory not found: {world_dir}. "
            "Place a valid Minecraft 1.16.5 world folder there."
        )

    world_name = os.path.basename(world_dir)  # e.g. "world"
    zip_dir = tempfile.mkdtemp(prefix="minerl_world_")
    zip_path = os.path.join(zip_dir, f"{world_name}.zip")
    staging_dir = os.path.join(tempfile.mkdtemp(prefix="minerl_world_stage_"), world_name)

    logger.info("Preparing world zip: %s -> %s", world_dir, zip_path)

    shutil.copytree(
        world_dir,
        staging_dir,
        ignore=shutil.ignore_patterns("session.lock"),
    )
    _sanitize_world_staging(staging_dir)

    import time
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(staging_dir):
            for fname in files:
                abs_path = os.path.join(root, fname)
                rel_path = os.path.relpath(abs_path, staging_dir).replace(os.sep, "/")
                # Verbatim "./saves/<world>/..." so that:
                #   - listZip()[0].split("/")[2] == world_name   (Java reads index 2)
                #   - Java's new File(dir, "./saves/...") -> dir/saves/...  (where loadWorld reads)
                # NOTE: zf.write() would run os.path.normpath and strip the "./",
                #       so the name must be set verbatim through ZipInfo.
                arc_name = f"./saves/{world_name}/{rel_path}"
                zi = zipfile.ZipInfo(arc_name, date_time=time.localtime(os.path.getmtime(abs_path))[:6])
                zi.compress_type = zipfile.ZIP_DEFLATED
                with open(abs_path, "rb") as fh:
                    zf.writestr(zi, fh.read())

    WORLD_ZIP = zip_path
    logger.info("World zip ready: %s (WORLD_ZIP set)", zip_path)
    return zip_path


# Prepare the world zip at import time so it's available when the env is created
try:
    prepare_world_zip(WORLD_DIR)
except FileNotFoundError as e:
    logger.warning("Could not prepare world zip: %s", e)
    logger.warning("The environment will generate a random world instead.")


# ===========================================================================
#  1a. Custom Block Grid Observation Handler
# ===========================================================================

class BlockGridHandler(Handler):
    """
    Injects an <ObservationFromGrid> element into the Malmo mission XML.

    This makes the Minecraft server report block types in a grid around the
    agent each tick.  The data appears in the raw Malmo JSON observation
    under the key given by *grid_name*.  Because this is a plain Handler
    (not a TranslationHandler), the data does NOT automatically appear in
    the gym observation dict -- we extract it manually in the wrapper.

    The grid is offset downward (negative y) to reach ground level when the
    player is riding a horse (~2 blocks above the ground block).
    """

    def __init__(
        self,
        grid_name: str = "floor_grid",
        min_x: int = -2, max_x: int = 2,
        min_y: int = -4, max_y: int = 0,
        min_z: int = -2, max_z: int = 2,
    ):
        self.grid_name = grid_name
        self.min_x, self.max_x = min_x, max_x
        self.min_y, self.max_y = min_y, max_y
        self.min_z, self.max_z = min_z, max_z

    def to_string(self) -> str:
        return f"block_grid_{self.grid_name}"

    def xml_template(self) -> str:
        return (
            f'<ObservationFromGrid>'
            f'<Grid name="{self.grid_name}">'
            f'<min x="{self.min_x}" y="{self.min_y}" z="{self.min_z}"/>'
            f'<max x="{self.max_x}" y="{self.max_y}" z="{self.max_z}"/>'
            f'</Grid>'
            f'</ObservationFromGrid>'
        )


class HorseSpawnDecorator(Handler):
    """
    Emits a DrawingDecorator with nested DrawEntity XML.

    The stock DrawingDecorator handler HTML-escapes inner XML under Jinja2
    autoescape, which prevents Malmo from parsing DrawEntity children.
    """

    MALMO_NS = "http://ProjectMalmo.microsoft.com"

    def __init__(self, draw_entity_xml: str):
        self.draw_entity_xml = draw_entity_xml

    def to_string(self) -> str:
        return "horse_spawn_decorator"

    def xml_template(self) -> str:
        return "<DrawingDecorator></DrawingDecorator>"

    def xml(self) -> str:
        return f"<DrawingDecorator>{self.draw_entity_xml}</DrawingDecorator>"


# ===========================================================================
#  1b. Environment Specification (herobraine)
# ===========================================================================

class HorseRaceEnvSpec(HumanControlEnvSpec):
    """
    Defines the Minecraft environment for the horse race task via MineRL's
    herobraine system.

    This spec:
      - Loads a custom world from disk (FileWorldGenerator)
      - Spawns a pre-tamed, saddled horse in front of the agent
      - Places the agent at the configured spawn coordinates
      - Provides only the POV (first-person RGB) observation
      - Exposes the standard human-like action set (keyboard + camera)
    """

    def __init__(self):
        super().__init__(
            name="HorseRace-v0",
            max_episode_steps=MAX_EPISODE_STEPS,
            resolution=NATIVE_RES,
            fov_range=[70, 70],
            gamma_range=[2, 2],
            guiscale_range=[1, 1],
            cursor_size_range=[16, 16],
        )

    # -- World generation ------------------------------------------------

    def create_server_world_generators(self) -> List[Handler]:
        """No world generator needed — the world is loaded via LoadWorldFile.

        MineRL v1.0.2's Java mod (EnvServer) ignores FileWorldGenerator XML
        entirely.  Instead, the world is injected via the AgentStart handler
        using <LoadWorldFile>, which the Java side reads and loads via
        ReplaySender.loadWorldFromZip().
        """
        return []

    def create_server_decorators(self) -> List[Handler]:
        """
        Spawn a tamed, saddled horse directly in front of the agent.

        Uses Malmo XML DrawEntity to place the horse at HORSE_X/Y/Z.
        EnvServer configures tame, saddle, variant, health, and attributes
        (see HORSE_* constants) after spawn.
        """
        draw_entity = (
            f'<DrawEntity xmlns="{HorseSpawnDecorator.MALMO_NS}" '
            f'x="{HORSE_X}" y="{HORSE_Y}" z="{HORSE_Z}" '
            f'yaw="{SPAWN_YAW}" type="Horse"/>'
        )
        return [HorseSpawnDecorator(draw_entity)]

    def create_server_initial_conditions(self) -> List[Handler]:
        """Set the world to noon and disable hostile mob spawning."""
        return [
            handlers.TimeInitialCondition(
                allow_passage_of_time=False,
            ),
            handlers.SpawningInitialCondition(
                allow_spawning=False,
            ),
        ]

    def create_server_quit_producers(self) -> List[Handler]:
        """End the episode on timeout or agent finish."""
        from minerl.herobraine.hero import mc
        return [
            handlers.ServerQuitFromTimeUp(
                MAX_EPISODE_STEPS * mc.MS_PER_STEP
            ),
            handlers.ServerQuitWhenAnyAgentFinishes(),
        ]

    # -- Agent -----------------------------------------------------------

    def create_agent_start(self) -> List[Handler]:
        """Spawn the agent at the configured coordinates and load the custom world."""
        agent_start = super().create_agent_start() + [
            AgentStartPlacement(
                x=SPAWN_X,
                y=SPAWN_Y,
                z=SPAWN_Z,
                yaw=SPAWN_YAW,
            ),
            DoneOnDeath(),
        ]
        # Inject the custom world via LoadWorldFile (zip path)
        if WORLD_ZIP is not None:
            agent_start.append(LoadWorldAgentStart(filename=WORLD_ZIP))
        return agent_start

    def create_observables(self) -> List[TranslationHandler]:
        """First-person camera view + agent location (for debugging position)."""
        return [
            handlers.POVObservation(self.resolution),
            handlers.ObservationFromCurrentLocation(),  # Adds agent position, yaw, pitch
        ]

    def create_actionables(self) -> List[TranslationHandler]:
        """Human-like keyboard + camera actions, with 'sneak' removed.

        On a horse, sneak (Left Shift) dismounts the rider. Rather than merely
        force-zeroing it in the wrapper, we drop the handler entirely so 'sneak'
        is not part of the underlying action space at all.
        """
        def _is_sneak(h) -> bool:
            ident = getattr(h, "command", None)
            if ident is None and hasattr(h, "to_string"):
                try:
                    ident = h.to_string()
                except Exception:
                    ident = None
            return ident == "sneak"

        return [h for h in super().create_actionables() if not _is_sneak(h)]

    def create_rewardables(self) -> List[TranslationHandler]:
        """Reward is computed externally in the Gym wrapper."""
        return []

    def create_agent_handlers(self) -> List[Handler]:
        """Include block grid observation for reward computation."""
        return [
            BlockGridHandler(
                grid_name="floor_grid",
                min_x=-2, max_x=2,
                min_y=-4, max_y=0,   # Covers ground level through horse body
                min_z=-2, max_z=2,
            ),
        ]

    def create_monitors(self) -> List[TranslationHandler]:
        return []

    # -- Misc required overrides -----------------------------------------

    def is_from_folder(self, folder: str) -> bool:
        return False

    def determine_success_from_rewards(self, rewards: list) -> bool:
        return sum(rewards) > 0

    def get_docstring(self):
        return self.__doc__


# Register the environment so gym.make() can find it
_horse_race_spec = HorseRaceEnvSpec()
_horse_race_spec.register()


# ===========================================================================
#  2. Gym Wrapper
# ===========================================================================

# Discrete action table — maps integer index to MineRL action dict overrides.
# Index 9 is a macro action handled specially in step() — not a simple override.
ACTION_TABLE = [
    # 0: No-op
    {},
    # 1: Forward
    {"forward": 1},
    # 2: Forward + Left
    {"forward": 1, "left": 1},
    # 3: Forward + Right
    {"forward": 1, "right": 1},
    # 4: Forward + Jump (single tick tap)
    {"forward": 1, "jump": 1},
    # 5: Camera — look left
    {"camera": np.array([0.0, -5.0])},
    # 6: Camera — look right
    {"camera": np.array([0.0, 5.0])},
    # 7: Camera — look up
    {"camera": np.array([-5.0, 0.0])},
    # 8: Camera — look down
    {"camera": np.array([5.0, 0.0])},
    # 9: Charge Jump (macro) — holds forward+jump for CHARGE_JUMP_TICKS ticks
    #    to clear 2-block fences. Handled specially in step().
    {"forward": 1, "jump": 1},
]

ACTION_CHARGE_JUMP = 9  # Index of the charge-jump macro action
NUM_ACTIONS = len(ACTION_TABLE)


class HorseRaceEnv(gym.Env):
    """
    Wraps the MineRL HorseRace-v0 environment with:
      - A simplified Discrete(9) action space
      - A preprocessed 64×64 RGB observation space
      - An automatic horse-mounting startup sequence on reset
      - A pluggable reward function (compute_reward)
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, visualize: bool = False, vis_size: Tuple[int, int] = NATIVE_RES, show_annotations: bool = True, video_path: Optional[str] = None):
        super().__init__()

        # Create the inner MineRL environment from our registered spec
        self._env = gym.make("HorseRace-v0")

        # --- Observation space ---
        # Preprocessed RGB image at OBS_WIDTH × OBS_HEIGHT
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(OBS_HEIGHT, OBS_WIDTH, 3),
            dtype=np.uint8,
        )

        # --- Action space ---
        # Simplified discrete actions (see ACTION_TABLE above)
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)

        # Internal state
        self._prev_obs = None

        # Raw last observation (MineRL pov) — used for visualization
        self._last_raw_obs = None

        # Visualization settings
        self._visualize = bool(visualize) and _CV2_AVAILABLE
        # vis_size provided as (width, height) to match NATIVE_RES ordering
        self._vis_size = tuple(vis_size)
        self._show_annotations = bool(show_annotations)
        self._video_path = video_path
        self._video_writer = None
        if self._visualize and self._video_path is not None and _CV2_AVAILABLE:
            # Initialize video writer lazily on first frame when we know frame size
            self._video_writer = None

        # Step tracking and position logging
        self._step_count = 0
        self._print_coords = True  # Set to False to disable position logging

        # --- Reward tracking state (reset each episode in reset()) ---
        self._init_reward_state()

    # -- Observation helpers ---------------------------------------------

    @staticmethod
    def _preprocess_obs(obs: dict) -> np.ndarray:
        """
        Extract the POV image from the MineRL observation dict and resize
        it to (OBS_HEIGHT, OBS_WIDTH, 3).
        """
        pov = obs["pov"]  # shape: (360, 640, 3) uint8
        img = Image.fromarray(pov)
        img = img.resize((OBS_WIDTH, OBS_HEIGHT), Image.BILINEAR)
        return np.array(img, dtype=np.uint8)

    # -- Reward state & helpers ------------------------------------------

    def _init_reward_state(self):
        """Initialise / reset all per-episode reward tracking variables."""
        self._next_checkpoint_idx = 1     # 0=Start already behind us; aim for CP_A
        self._last_pos = None             # (x, y, z) from previous step
        self._position_history = collections.deque(maxlen=STUCK_WINDOW)
        self._last_ground_block = "air"
        self._force_done = False          # Set True to end episode early

        # -- Timing / statistics state --
        self._episode_start_step = 0          # Set after mount in reset()
        self._lap_complete = False            # True when all checkpoints crossed
        # Step at which each checkpoint was crossed; None if not reached.
        # Index 0 = Start/Goal (lap finish), 1..5 = CP_A..CP_E
        self._checkpoint_times = [None] * NUM_CHECKPOINTS
        # Time (seconds) for each of the 6 segments; None if segment not completed
        self._segment_splits = [None] * NUM_CHECKPOINTS
        # Step when the agent entered the current segment (last checkpoint crossed)
        self._segment_enter_step = 0

    @staticmethod
    def _extract_position(obs: dict):
        """Return (x, y, z) from MineRL observation dict, or None."""
        try:
            loc = obs["location_stats"]
            return (float(loc["xpos"]), float(loc["ypos"]), float(loc["zpos"]))
        except Exception:
            return None

    @staticmethod
    def _extract_ground_block(obs: dict) -> str:
        """
        Return the block name directly below the horse from the grid
        observation.  Falls back to 'unknown' if unavailable.

        The grid is x=[-2..2], y=[-4..0], z=[-2..2] (sx=5, sy=5, sz=5).
        Malmo orders grid data as: for y ascending, then z ascending,
        then x ascending.  Flat index formula:
            idx = (y - min_y) * sz * sx + (z - min_z) * sx + (x - min_x)

        For the center ground block at (x=0, z=0, y=-HORSE_Y_OFFSET):
            y_off = -HORSE_Y_OFFSET - (-4) = 4 - HORSE_Y_OFFSET
            idx = y_off * 25 + 2 * 5 + 2
        """
        try:
            grid = obs["floor_grid"]
            y_off = 4 - HORSE_Y_OFFSET  # y=-HORSE_Y_OFFSET relative to min_y=-4
            idx = y_off * 25 + 2 * 5 + 2  # center x and z
            return str(grid[idx])
        except Exception:
            return "unknown"

    @staticmethod
    def _checkpoint_crossed(pos, prev_pos, cp) -> bool:
        """
        Check if the agent crossed a checkpoint line between prev_pos and pos.

        Each checkpoint has an 'axis' ('x' or 'z') indicating which coordinate
        is constant along the gate line.  The agent crosses the gate when that
        coordinate changes sign relative to the gate value between two steps,
        AND the other coordinate is within the gate endpoints.
        """
        if prev_pos is None or pos is None:
            return False
        p1, p2 = cp["p1"], cp["p2"]
        if cp["axis"] == "z":
            gate_z = p1[1]  # constant Z
            x_min = min(p1[0], p2[0])
            x_max = max(p1[0], p2[0])
            # Check Z crossing
            if (prev_pos[2] - gate_z) * (pos[2] - gate_z) <= 0:
                # Check X within gate span
                if x_min <= pos[0] <= x_max:
                    return True
        else:  # axis == "x"
            gate_x = p1[0]  # constant X
            z_min = min(p1[1], p2[1])
            z_max = max(p1[1], p2[1])
            # Check X crossing
            if (prev_pos[0] - gate_x) * (pos[0] - gate_x) <= 0:
                # Check Z within gate span
                if z_min <= pos[2] <= z_max:
                    return True
        return False

    @staticmethod
    def _dist_to_checkpoint(pos, cp) -> float:
        """Euclidean XZ distance from pos to the midpoint of a checkpoint."""
        mid_x = (cp["p1"][0] + cp["p2"][0]) / 2.0
        mid_z = (cp["p1"][1] + cp["p2"][1]) / 2.0
        return np.sqrt((pos[0] - mid_x) ** 2 + (pos[2] - mid_z) ** 2)

    @staticmethod
    def _min_dist_to_any_checkpoint(pos) -> float:
        """Minimum XZ distance from pos to any checkpoint midpoint."""
        return min(
            HorseRaceEnv._dist_to_checkpoint(pos, cp) for cp in CHECKPOINTS
        )

    # -- Action helpers --------------------------------------------------

    def _get_noop_action(self) -> dict:
        """Return a no-op action dict compatible with the inner env."""
        return self._env.action_space.no_op()

    def _map_action(self, action_index: int) -> dict:
        """
        Convert a discrete action index into a full MineRL action dict.

        Starts from a no-op action and applies the overrides from
        ACTION_TABLE[action_index].  Always forces sneak=0 to prevent
        the agent from dismounting the horse.
        """
        act = self._get_noop_action()
        overrides = ACTION_TABLE[action_index]
        for key, value in overrides.items():
            act[key] = value
        # Prevent dismounting: sneak (Shift) dismounts the horse in MC
        if "sneak" in act:
            act["sneak"] = 0
        return act

    # -- Horse mounting --------------------------------------------------

    def _log_position(self, obs: dict, step: int = 0) -> None:
        """
        Extract and print agent position from observation.

        The ObservationFromCurrentLocation handler adds position/rotation data
        to the observation dict. Tries multiple field name variations since
        different MineRL versions use different names.
        """
        if not self._print_coords or obs is None:
            return
        pos = obs['location_stats']
        try:
            # Try multiple field name variations (different MineRL versions)
            x = pos['xpos']
            y = pos['ypos']
            z = pos['zpos']
            yaw = pos['yaw']
            pitch = pos['pitch']
            logger.info(
                f"Step {step:5d} | Pos: X={x:7.2f} Y={y:6.2f} Z={z:7.2f} | "
                f"Yaw={yaw:6.1f}° Pitch={pitch:6.1f}°"
            )
        except Exception as e:
            logger.debug(f"Could not extract position: {e}")

    # -- Horse mounting --------------------------------------------------

    def _mount_horse(self):
        """
        Execute the fixed startup sequence to mount the horse and orient
        the agent toward the race track.

        This runs inside reset() so PPO never sees these setup steps.

        Sequence:
          1. Wait a few ticks for the world to settle
          2. Send 'use' (right-click) to mount the nearby horse
          3. Rotate camera to face the start of the race track
        """
        # The horse is spawned natively, server-side, by the DrawEntity world
        # decorator (see create_server_decorators -> EnvServer.spawnDrawEntity),
        # so there is nothing to spawn here. We only need to walk up to it and
        # mount it below.

        # Let the world settle for a few ticks
        noop = self._get_noop_action()
        pre_mount_y = None
        for _ in range(5):
            obs, _, done, _ = self._env.step(noop)
            if done:
                return obs
            try:
                pre_mount_y = obs["location_stats"]["ypos"]
            except (KeyError, TypeError):
                pass

        # Walk toward the horse (spawned HORSE_DISTANCE blocks ahead along facing)
        forward = self._map_action(1)
        for _ in range(HORSE_DISTANCE + 2):
            obs, _, done, _ = self._env.step(forward)
            if done:
                return obs

        # Mount the horse (right-click / 'use' action)
        mount_action = self._get_noop_action()
        mount_action["use"] = 1
        for _ in range(5):
            obs, _, done, _ = self._env.step(mount_action)
            if done:
                return obs

        # Small pause to let mounting animation complete
        for _ in range(5):
            obs, _, done, _ = self._env.step(noop)
            if done:
                return obs

        try:
            post_mount_y = obs["location_stats"]["ypos"]
            if pre_mount_y is not None and post_mount_y <= pre_mount_y + 0.3:
                logger.warning(
                    "Horse mount may have failed (ypos %.2f -> %.2f). "
                    "Check that the horse spawned at (%.1f, %.1f, %.1f).",
                    pre_mount_y,
                    post_mount_y,
                    HORSE_X,
                    HORSE_Y,
                    HORSE_Z,
                )
        except (KeyError, TypeError):
            pass

        # TODO: Rotate camera to face the race track start direction.
        #       Adjust the yaw delta below based on your track layout.
        #       Positive camera[1] = look right, negative = look left.
        # rotate_action = self._get_noop_action()
        # rotate_action["camera"] = np.array([0.0, 90.0])
        # obs, _, done, _ = self._env.step(rotate_action)

        return obs

    # -- Gym API ---------------------------------------------------------

    def reset(self):
        """
        Reset the environment, mount the horse, and return the first
        preprocessed observation.
        """
        obs = self._env.reset()

        # Execute the horse-mounting startup sequence
        obs_after_mount = self._mount_horse()
        if obs_after_mount is not None:
            obs = obs_after_mount

        # Keep the raw MineRL observation for visualization
        try:
            self._last_raw_obs = obs
        except Exception:
            self._last_raw_obs = None

        # Log initial position
        self._step_count = 0
        self._log_position(obs, step=0)

        # Reset reward tracking for new episode
        self._init_reward_state()
        # Record the step at which gameplay begins (after mount sequence)
        self._episode_start_step = self._step_count
        self._segment_enter_step = self._step_count
        pos = self._extract_position(obs)
        if pos is not None:
            self._last_pos = pos
            self._position_history.append(pos)

        processed = self._preprocess_obs(obs)
        self._prev_obs = processed
        return processed

    def step(self, action: int):
        """
        Take a step in the environment.

        Args:
            action: Integer index into ACTION_TABLE.

        Returns:
            Tuple of (observation, reward, done, info).
        """
        # -- Handle charge-jump macro (action 9) --
        if action == ACTION_CHARGE_JUMP:
            return self._step_charge_jump()

        minerl_action = self._map_action(action)
        result = self._env.step(minerl_action)
        if len(result) == 5:
            obs, _minerl_reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, _minerl_reward, done, info = result

        # Increment step counter and log position
        self._step_count += 1
        self._log_position(obs, step=self._step_count)

        if self._visualize:
            self.render()
        try:
            self._last_raw_obs = obs
        except Exception:
            self._last_raw_obs = None
        processed = self._preprocess_obs(obs)

        reward = self.compute_reward(
            obs=processed,
            prev_obs=self._prev_obs,
            action=action,
            raw_obs=obs,
        )

        done = done or self._force_done
        self._prev_obs = processed
        return processed, reward, done, info

    def _step_charge_jump(self):
        """
        Execute the charge-jump macro: hold forward+jump for
        CHARGE_JUMP_TICKS consecutive ticks, accumulating reward.

        Returns the same (obs, reward, done, info) tuple as step().
        """
        total_reward = 0.0
        jump_action = self._map_action(ACTION_CHARGE_JUMP)

        for tick in range(CHARGE_JUMP_TICKS):
            result = self._env.step(jump_action)
            if len(result) == 5:
                obs, _, terminated, truncated, info = result
                done = terminated or truncated
            else:
                obs, _, done, info = result

            self._step_count += 1
            self._log_position(obs, step=self._step_count)

            try:
                self._last_raw_obs = obs
            except Exception:
                self._last_raw_obs = None
            processed = self._preprocess_obs(obs)

            total_reward += self.compute_reward(
                obs=processed,
                prev_obs=self._prev_obs,
                action=ACTION_CHARGE_JUMP,
                raw_obs=obs,
            )
            self._prev_obs = processed

            if done or self._force_done:
                done = True
                break

        if self._visualize:
            self.render()

        return processed, total_reward, done, info


    def render(self, mode: str = "human", return_frame: bool = False):
        """Render or return a visual frame using OpenCV.

        When visualization is enabled and OpenCV is available, this will show
        a window (non-blocking) and optionally return the BGR frame.
        """
        if not _CV2_AVAILABLE:
            return None

        raw = self._last_raw_obs
        if raw is None or "pov" not in raw:
            return None

        frame_rgb = raw["pov"]
        # Ensure numpy array
        frame = np.array(frame_rgb, dtype=np.uint8)

        # Convert RGB -> BGR for OpenCV
        try:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        except Exception:
            # If conversion fails (odd shapes), fall back to using as-is
            bgr = frame

        # Resize to requested visualization size (vis_size is (width, height))
        try:
            width, height = self._vis_size
            bgr = cv2.resize(bgr, (int(width), int(height)), interpolation=cv2.INTER_LINEAR)
        except Exception:
            pass

        # Annotations (optional)
        if self._show_annotations and _CV2_AVAILABLE:
            text = "HorseRaceEnv"
            cv2.putText(bgr, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        # Initialize video writer lazily if requested
        if self._video_path is not None and self._video_writer is None and _CV2_AVAILABLE:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            h, w = bgr.shape[:2]
            self._video_writer = cv2.VideoWriter(self._video_path, fourcc, 20.0, (w, h))

        if self._video_writer is not None:
            # VideoWriter expects BGR frames
            self._video_writer.write(bgr)

        if self._visualize and _CV2_AVAILABLE:
            try:
                cv2.imshow("HorseRace", bgr)
                cv2.waitKey(1)
            except Exception:
                # In headless contexts this may fail; ignore to keep training running
                pass

        if return_frame:
            return bgr

    def compute_reward(
        self,
        obs: np.ndarray,
        prev_obs: np.ndarray,
        action: int,
        raw_obs: dict,
    ) -> float:
        """
        Multi-signal reward function for horse racing.

        Signals (all values from tunable constants at top of file):
          - Checkpoint crossing          (+REWARD_CHECKPOINT / +REWARD_LAP_COMPLETE)
          - Progress toward next CP      (+REWARD_PROGRESS * delta_distance)
          - On grass_path / dirt_path     (+REWARD_ON_PATH per step)
          - On gold_block (speed boost)   (+REWARD_GOLD_BLOCK)
          - On spruce_slab (bridge)       (+REWARD_SPRUCE_SLAB)
          - On soul_sand                  (PENALTY_SOUL_SAND per step)
          - In water                      (PENALTY_WATER per step)
          - In cobweb                     (PENALTY_COBWEB per step)
          - Off-course (grass_block)      (PENALTY_OFF_COURSE per step)
          - Time penalty                  (PENALTY_TIME per step)
          - Stuck too long                (PENALTY_STUCK + episode ends)
          - Far off-course                (PENALTY_FAR_OFF_COURSE + episode ends)
          - Wrong direction (backward)    (PENALTY_WRONG_DIRECTION per step)

        Args:
            obs:      Current preprocessed observation (H×W×3 uint8).
            prev_obs: Previous preprocessed observation.
            action:   The discrete action index that was taken.
            raw_obs:  The raw MineRL observation dict (contains location_stats
                      and, if available, floor_grid).

        Returns:
            A float reward value.
        """
        reward = 0.0
        pos = self._extract_position(raw_obs)
        ground_block = self._extract_ground_block(raw_obs)
        self._last_ground_block = ground_block

        # ---- 1. Checkpoint crossing ------------------------------------
        if pos is not None and self._last_pos is not None:
            target_cp = CHECKPOINTS[self._next_checkpoint_idx]
            if self._checkpoint_crossed(pos, self._last_pos, target_cp):
                # Record timestamp and segment split
                self._checkpoint_times[self._next_checkpoint_idx] = self._step_count
                segment_steps = self._step_count - self._segment_enter_step
                # Map checkpoint index to segment index:
                # Segment 0 = Start→CP_A (entered at start, completed at CP_A idx=1)
                # ...
                # Segment 5 = CP_E→Finish (entered at CP_E, completed at Start idx=0)
                if self._next_checkpoint_idx == 0:
                    seg_idx = NUM_CHECKPOINTS - 1  # last segment
                else:
                    seg_idx = self._next_checkpoint_idx - 1
                self._segment_splits[seg_idx] = segment_steps / TICKS_PER_SECOND
                self._segment_enter_step = self._step_count

                if self._next_checkpoint_idx == 0:
                    # Crossed start/goal → lap complete
                    reward += REWARD_LAP_COMPLETE
                    self._lap_complete = True
                    logger.info(">>> LAP COMPLETE!")
                else:
                    reward += REWARD_CHECKPOINT
                    logger.info(
                        f">>> Checkpoint {target_cp['name']} crossed! "
                        f"(+{REWARD_CHECKPOINT})"
                    )
                # Advance to next checkpoint (wrap around for lap)
                self._next_checkpoint_idx = (
                    (self._next_checkpoint_idx + 1) % NUM_CHECKPOINTS
                )

        # ---- 2. Progress toward next checkpoint -----------------------
        if pos is not None and self._last_pos is not None:
            target_cp = CHECKPOINTS[self._next_checkpoint_idx]
            prev_dist = self._dist_to_checkpoint(self._last_pos, target_cp)
            curr_dist = self._dist_to_checkpoint(pos, target_cp)
            delta = prev_dist - curr_dist  # positive = getting closer
            reward += REWARD_PROGRESS * delta

        # ---- 2b. Wrong-direction penalty (going backward) --------------
        #  Detect if the agent is heading backward by checking whether it
        #  is getting closer to the *previous* checkpoint rather than the
        #  next one.  This catches the agent trying to short-cut by
        #  running the loop in reverse.
        if pos is not None and self._last_pos is not None:
            prev_cp_idx = (self._next_checkpoint_idx - 1) % NUM_CHECKPOINTS
            prev_cp = CHECKPOINTS[prev_cp_idx]
            prev_dist_back = self._dist_to_checkpoint(self._last_pos, prev_cp)
            curr_dist_back = self._dist_to_checkpoint(pos, prev_cp)
            if curr_dist_back < prev_dist_back - 0.5:
                # Agent is moving toward the checkpoint it already passed
                reward += PENALTY_WRONG_DIRECTION

        # ---- 3. Block-type rewards / penalties -------------------------
        if ground_block in ("grass_path", "dirt_path"):
            reward += REWARD_ON_PATH
        elif ground_block == "gold_block":
            reward += REWARD_GOLD_BLOCK
        elif ground_block in ("spruce_slab",):
            reward += REWARD_SPRUCE_SLAB
        elif ground_block == "soul_sand":
            reward += PENALTY_SOUL_SAND
        elif ground_block in ("water", "flowing_water"):
            reward += PENALTY_WATER
        elif ground_block == "cobweb":
            reward += PENALTY_COBWEB
        elif ground_block == "grass_block":
            reward += PENALTY_OFF_COURSE

        # ---- 4. Time penalty -------------------------------------------
        reward += PENALTY_TIME

        # ---- 5. Stuck detection ----------------------------------------
        if pos is not None:
            self._position_history.append(pos)
            if len(self._position_history) >= STUCK_WINDOW:
                oldest = self._position_history[0]
                dx = pos[0] - oldest[0]
                dz = pos[2] - oldest[2]
                displacement = np.sqrt(dx * dx + dz * dz)
                if displacement < STUCK_MIN_DISPLACEMENT:
                    reward += PENALTY_STUCK
                    self._force_done = True
                    logger.info(
                        f">>> STUCK detected (moved {displacement:.2f} blocks "
                        f"in {STUCK_WINDOW} steps). Ending episode."
                    )

        # ---- 6. Far off-course detection --------------------------------
        if pos is not None:
            min_dist = self._min_dist_to_any_checkpoint(pos)
            if min_dist > OFF_COURSE_MAX_DIST:
                reward += PENALTY_FAR_OFF_COURSE
                self._force_done = True
                logger.info(
                    f">>> FAR OFF COURSE (nearest CP: {min_dist:.1f} blocks). "
                    f"Ending episode."
                )

        # ---- Update state for next step --------------------------------
        if pos is not None:
            self._last_pos = pos

        return reward

    def close(self):
        """Clean up the inner MineRL environment."""
        self._env.close()
        # Release video writer and destroy windows if used
        if getattr(self, "_video_writer", None) is not None:
            try:
                self._video_writer.release()
            except Exception:
                pass
            self._video_writer = None

        if _CV2_AVAILABLE:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass


# ===========================================================================
#  3. Environment Factory
# ===========================================================================

def make_env():
    """
    Factory function that returns a callable (thunk) for creating a
    HorseRaceEnv instance.  Used by SB3's DummyVecEnv.
    """
    def _init():
        return HorseRaceEnv(visualize=VISUALIZE)
    return _init


# ===========================================================================
#  4. Reward Tracking Callback
# ===========================================================================

class RewardTrackingCallback(BaseCallback):
    """
    A stable-baselines3 callback that records per-episode statistics
    during training.

    Tracked metrics (access after training):
      - ``episode_rewards``   — total reward per episode
      - ``episode_durations`` — total episode time in seconds (None if lap not completed)
      - ``segment_times``     — list of 6-element lists; each element is the time
                                in seconds for that segment, or None if not completed
      - ``completion_rates``  — fraction of checkpoints reached (0.0–1.0)
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards: List[float] = []
        self.episode_durations: List[Optional[float]] = []
        self.segment_times: List[List[Optional[float]]] = []
        self.completion_rates: List[float] = []
        self._current_rewards: List[float] = []

    def _on_training_start(self):
        """Initialise per-environment reward accumulators."""
        n_envs = self.training_env.num_envs
        self._current_rewards = [0.0] * n_envs

    def _get_inner_env(self, vec_env_idx: int):
        """Navigate through SB3 VecEnv wrappers to reach HorseRaceEnv."""
        # VecTransposeImage wraps DummyVecEnv; DummyVecEnv.envs is a list
        venv = self.training_env
        # Walk through VecEnvWrapper layers to find DummyVecEnv
        while hasattr(venv, "venv"):
            venv = venv.venv
        # DummyVecEnv stores envs as a list
        return venv.envs[vec_env_idx]

    def _on_step(self) -> bool:
        """Called after each environment step."""
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]

        for i, (reward, done) in enumerate(zip(rewards, dones)):
            self._current_rewards[i] += reward
            if done:
                self.episode_rewards.append(self._current_rewards[i])
                self._current_rewards[i] = 0.0

                # --- Collect timing / completion stats from inner env ---
                try:
                    inner_env = self._get_inner_env(i)
                    # Episode duration (seconds)
                    if inner_env._lap_complete:
                        total_steps = (inner_env._step_count
                                       - inner_env._episode_start_step)
                        self.episode_durations.append(
                            total_steps / TICKS_PER_SECOND
                        )
                    else:
                        self.episode_durations.append(None)

                    # Segment splits (already in seconds)
                    self.segment_times.append(
                        list(inner_env._segment_splits)
                    )

                    # Completion rate: how many checkpoints were reached
                    reached = sum(
                        1 for t in inner_env._checkpoint_times if t is not None
                    )
                    self.completion_rates.append(reached / NUM_CHECKPOINTS)

                except Exception as e:
                    logger.warning(
                        f"Could not read stats from inner env: {e}"
                    )
                    self.episode_durations.append(None)
                    self.segment_times.append([None] * NUM_CHECKPOINTS)
                    self.completion_rates.append(0.0)

                if self.verbose > 0:
                    ep_num = len(self.episode_rewards)
                    dur = self.episode_durations[-1]
                    rate = self.completion_rates[-1]
                    dur_str = f"{dur:.1f}s" if dur is not None else "DNF"
                    logger.info(
                        f"Episode {ep_num} — "
                        f"Reward: {self.episode_rewards[-1]:.2f}  "
                        f"Duration: {dur_str}  "
                        f"Completion: {rate:.0%}"
                    )

        return True  # Continue training


# ===========================================================================
#  5. Plotting
# ===========================================================================

def plot_rewards(
    episode_rewards: List[float],
    window: int = 20,
    save_path: str = REWARD_PLOT_PATH,
):
    """
    Plot the agent's total reward per episode over the course of training.

    Generates two lines:
      - Raw episode reward (translucent)
      - Rolling average (smoothed, solid)

    Args:
        episode_rewards: List of total rewards, one per completed episode.
        window:          Rolling-average window size.
        save_path:       File path to save the plot image.
    """
    if not episode_rewards:
        logger.warning("No episode rewards to plot.")
        return

    episodes = np.arange(1, len(episode_rewards) + 1)
    rewards = np.array(episode_rewards)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, rewards, alpha=0.3, color="steelblue", label="Episode Reward")

    # Rolling average
    if len(rewards) >= window:
        rolling = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(
            episodes[window - 1:],
            rolling,
            color="steelblue",
            linewidth=2,
            label=f"Rolling Avg (window={window})",
        )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.set_title("Horse Race PPO — Training Performance")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Reward plot saved to {save_path}")


def plot_race_time(
    episode_durations: List[Optional[float]],
    window: int = 20,
    save_path: str = RACE_TIME_PLOT_PATH,
):
    """
    Plot the total time to complete the full lap per episode.

    Episodes where the lap was not completed (None) are shown as red
    '×' markers at a visual sentinel position. Completed episodes are
    plotted as dots with a rolling-average line.  Gaps are preserved
    between completed data points when separated by DNF episodes.

    Args:
        episode_durations: Time in seconds per episode, or None for DNF.
        window:            Rolling-average window size.
        save_path:         File path to save the plot image.
    """
    if not episode_durations:
        logger.warning("No episode durations to plot.")
        return

    episodes = np.arange(1, len(episode_durations) + 1)
    # Build arrays with NaN for DNF episodes so matplotlib leaves gaps
    durations = np.array(
        [d if d is not None else np.nan for d in episode_durations],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot completed episodes (scatter so gaps are visible)
    completed_mask = ~np.isnan(durations)
    if completed_mask.any():
        ax.scatter(
            episodes[completed_mask],
            durations[completed_mask],
            s=12, alpha=0.5, color="steelblue", label="Completed",
        )

    # Plot DNF markers
    dnf_mask = np.isnan(durations)
    if dnf_mask.any():
        # Place DNF markers at the top of the plot area
        sentinel = MAX_EPISODE_STEPS / TICKS_PER_SECOND
        ax.scatter(
            episodes[dnf_mask],
            [sentinel] * int(dnf_mask.sum()),
            s=30, marker="x", color="crimson", alpha=0.6,
            label="Did Not Finish",
        )

    # Rolling average over completed times only (with NaN-aware method)
    if completed_mask.sum() >= window:
        # Compute rolling mean manually, skipping NaNs
        rolling_vals = []
        rolling_eps = []
        for j in range(len(durations)):
            win_start = max(0, j - window + 1)
            win_slice = durations[win_start:j + 1]
            valid = win_slice[~np.isnan(win_slice)]
            if len(valid) >= window:
                rolling_vals.append(np.mean(valid[-window:]))
                rolling_eps.append(episodes[j])
        if rolling_vals:
            ax.plot(
                rolling_eps, rolling_vals,
                color="steelblue", linewidth=2,
                label=f"Rolling Avg (window={window})",
            )

    ax.set_xlabel("Episode")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Horse Race PPO — Lap Completion Time")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Race time plot saved to {save_path}")


# Segment labels for the 6 track sections
_SEGMENT_LABELS = [
    "Start → CP_A",
    "CP_A → CP_B",
    "CP_B → CP_C",
    "CP_C → CP_D",
    "CP_D → CP_E",
    "CP_E → Finish",
]


def plot_segment_times(
    segment_times: List[List[Optional[float]]],
    window: int = 20,
    save_path: str = SEGMENT_TIME_PLOT_PATH,
):
    """
    Plot a 2×3 grid of subplots, one per track segment.

    Each subplot shows the time (seconds) to complete that segment per
    episode.  Episodes where the segment was not completed are left as
    gaps (NaN) so the line does not connect across them.

    Args:
        segment_times: List of 6-element lists (one per episode).
                       Each element is seconds or None.
        window:        Rolling-average window size.
        save_path:     File path to save the combined plot image.
    """
    if not segment_times:
        logger.warning("No segment times to plot.")
        return

    n_episodes = len(segment_times)
    episodes = np.arange(1, n_episodes + 1)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    axes_flat = axes.flatten()

    for seg_idx in range(NUM_CHECKPOINTS):
        ax = axes_flat[seg_idx]
        # Extract this segment's times across all episodes
        times = np.array(
            [
                ep[seg_idx] if ep[seg_idx] is not None else np.nan
                for ep in segment_times
            ],
            dtype=float,
        )

        completed_mask = ~np.isnan(times)
        n_completed = int(completed_mask.sum())

        # Scatter completed times (gaps where NaN)
        if n_completed > 0:
            ax.scatter(
                episodes[completed_mask],
                times[completed_mask],
                s=10, alpha=0.5, color="teal",
            )

        # Rolling average (NaN-aware, same approach as race time)
        if n_completed >= window:
            rolling_vals = []
            rolling_eps = []
            for j in range(n_episodes):
                win_start = max(0, j - window + 1)
                win_slice = times[win_start:j + 1]
                valid = win_slice[~np.isnan(win_slice)]
                if len(valid) >= window:
                    rolling_vals.append(np.mean(valid[-window:]))
                    rolling_eps.append(episodes[j])
            if rolling_vals:
                ax.plot(
                    rolling_eps, rolling_vals,
                    color="teal", linewidth=2,
                )

        label = _SEGMENT_LABELS[seg_idx] if seg_idx < len(_SEGMENT_LABELS) else f"Segment {seg_idx}"
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("Time (s)", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)

        # Annotation: completed count
        ax.text(
            0.98, 0.95,
            f"Completed: {n_completed}/{n_episodes}",
            transform=ax.transAxes,
            fontsize=7, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
        )

    # Common X label on bottom row
    for ax in axes_flat[3:]:
        ax.set_xlabel("Episode", fontsize=9)

    fig.suptitle("Horse Race PPO — Segment Split Times", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Segment times plot saved to {save_path}")


def plot_completion_rate(
    completion_rates: List[float],
    window: int = 20,
    save_path: str = COMPLETION_RATE_PLOT_PATH,
):
    """
    Plot the fraction of checkpoints the agent reached per episode.

    Args:
        completion_rates: Values in [0.0, 1.0], one per episode.
        window:           Rolling-average window size.
        save_path:        File path to save the plot image.
    """
    if not completion_rates:
        logger.warning("No completion rates to plot.")
        return

    episodes = np.arange(1, len(completion_rates) + 1)
    rates = np.array(completion_rates)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        episodes, rates * 100,
        alpha=0.3, color="darkorange", label="Episode Completion %",
    )

    # Rolling average
    if len(rates) >= window:
        rolling = np.convolve(rates, np.ones(window) / window, mode="valid")
        ax.plot(
            episodes[window - 1:],
            rolling * 100,
            color="darkorange", linewidth=2,
            label=f"Rolling Avg (window={window})",
        )

    # Reference lines
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.axhline(50, color="gray", linestyle="--", linewidth=0.5)
    ax.axhline(100, color="gray", linestyle="--", linewidth=0.5)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Track Completion (%)")
    ax.set_ylim(-5, 105)
    ax.set_title("Horse Race PPO — Track Completion Rate")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Completion rate plot saved to {save_path}")


# ===========================================================================
#  6. Training Entrypoint
# ===========================================================================

def train(total_timesteps: int = TOTAL_TIMESTEPS):
    """
    Main training function.

    Creates the environment, configures PPO with a CNN policy, trains for
    the specified number of timesteps, then saves the model and plots the
    reward curve.

    Args:
        total_timesteps: Total number of environment steps to train for.
    """
    logger.info("Creating vectorised environment...")
    env = DummyVecEnv([make_env()])

    # VecTransposeImage converts observations from (H, W, C) to (C, H, W)
    # which is the format expected by PyTorch CNN policies.
    env = VecTransposeImage(env)

    logger.info("Initialising PPO with CnnPolicy...")

    # load horse_race_ppo.zip if it exists
    if os.path.exists("horse_race_ppo.zip"):
        model = PPO.load("horse_race_ppo.zip", env=env)
        logger.info("Loaded pre-trained model from horse_race_ppo.zip")
    else:
        logger.info("No pre-trained model found, training from scratch")
        model = PPO(
            policy="CnnPolicy",
            env=env,
            learning_rate=LEARNING_RATE,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            clip_range=CLIP_RANGE,
            verbose=1,
            # Uncomment and modify the line below to use a custom visual encoder:
            # policy_kwargs=dict(
            #     features_extractor_class=MyCustomEncoder,
            #     features_extractor_kwargs=dict(features_dim=512),
            # ),
        )

    # Set up reward tracking
    reward_callback = RewardTrackingCallback(verbose=1)

    logger.info(f"Starting training for {total_timesteps:,} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=reward_callback,
    )

    # Save the trained model
    model.save(MODEL_SAVE_PATH)
    logger.info(f"Model saved to {MODEL_SAVE_PATH}")

    # Plot the training reward curve
    plot_rewards(reward_callback.episode_rewards)

    # Plot race time per episode
    plot_race_time(reward_callback.episode_durations)

    # Plot segment split times (2×3 grid)
    plot_segment_times(reward_callback.segment_times)

    # Plot track completion rate
    plot_completion_rate(reward_callback.completion_rates)

    # Clean up
    env.close()
    logger.info("Training complete.")


# ===========================================================================
#  7. Main
# ===========================================================================

if __name__ == "__main__":
    train(total_timesteps=TOTAL_TIMESTEPS)
