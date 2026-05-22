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
from typing import List

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
from minerl.herobraine.hero.handlers.server.world import DrawingDecorator
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
SPAWN_Y = 100.0     # Adjust Y to match your world's ground level at (0, Z=0)
SPAWN_Z = -149.0
SPAWN_YAW = 180.0    # Facing +Z direction (toward the horse / race track start)

# -- Horse spawn (directly in front of agent) --
HORSE_X = SPAWN_X
HORSE_Y = SPAWN_Y       # Same ground level as agent
HORSE_Z = SPAWN_Z+2        # 2 blocks ahead of the agent

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

    logger.info("Preparing world zip: %s -> %s", world_dir, zip_path)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Walk the world directory and add all files under
        # the prefix structure: _/saves/<world_name>/...
        # Skip session.lock to prevent Minecraft from rejecting the world.
        prefix = os.path.join("_", "saves", world_name)
        for root, dirs, files in os.walk(world_dir):
            for fname in files:
                if fname == "session.lock":
                    continue
                abs_path = os.path.join(root, fname)
                rel_path = os.path.relpath(abs_path, world_dir)
                arc_name = os.path.join(prefix, rel_path)
                zf.write(abs_path, arc_name)

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
        The horse is pre-tamed (Tame:1b) and given a saddle so the agent can
        mount it immediately.
        """
        return [
            DrawingDecorator(
                f'<DrawEntity x="{HORSE_X}" y="{HORSE_Y}" z="{HORSE_Z}" yaw="{SPAWN_YAW}"'
                f'type="minecraft:horse">'
                f'<NBTData>{{Tame:1b, SaddleItem:{{id:"minecraft:saddle",Count:1b}}, Attributes:[{{Name:"minecraft:generic.movement_speed",Base:0.2}}, {{Name:"minecraft:horse.jump_strength",Base:.85}}, {{Name:"minecraft:generic.max_health",Base:20.0}}],  Health:20.0f,  Variant:1029}}</NBTData>'
                f'</DrawEntity>'
            )
        ]

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
        """Standard human-like keyboard + camera actions."""
        return super().create_actionables()

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
        ACTION_TABLE[action_index].
        """
        act = self._get_noop_action()
        overrides = ACTION_TABLE[action_index]
        for key, value in overrides.items():
            act[key] = value
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
        # Let the world settle for a few ticks
        noop = self._get_noop_action()
        for _ in range(5):
            obs, _, done, _ = self._env.step(noop)
            if done:
                return obs

        # Mount the horse (right-click / 'use' action)
        mount_action = self._get_noop_action()
        mount_action["use"] = 1
        for _ in range(3):
            obs, _, done, _ = self._env.step(mount_action)
            if done:
                return obs

        # Small pause to let mounting animation complete
        for _ in range(5):
            obs, _, done, _ = self._env.step(noop)
            if done:
                return obs

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
                if self._next_checkpoint_idx == 0:
                    # Crossed start/goal → lap complete
                    reward += REWARD_LAP_COMPLETE
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
    A stable-baselines3 callback that records the total reward for each
    completed episode during training.

    After training, access the rewards via `callback.episode_rewards`.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards: List[float] = []
        self._current_rewards: List[float] = []

    def _on_training_start(self):
        """Initialise per-environment reward accumulators."""
        n_envs = self.training_env.num_envs
        self._current_rewards = [0.0] * n_envs

    def _on_step(self) -> bool:
        """Called after each environment step."""
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]

        for i, (reward, done) in enumerate(zip(rewards, dones)):
            self._current_rewards[i] += reward
            if done:
                self.episode_rewards.append(self._current_rewards[i])
                self._current_rewards[i] = 0.0

                if self.verbose > 0:
                    logger.info(
                        f"Episode {len(self.episode_rewards)} — "
                        f"Total Reward: {self.episode_rewards[-1]:.2f}"
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

    # ------------------------------------------------------------------
    #  Visual Encoder Plug-in Point
    # ------------------------------------------------------------------
    #
    #  By default, PPO("CnnPolicy", ...) uses stable-baselines3's built-in
    #  NatureCNN as the visual feature extractor. If you want to use a
    #  different visual encoder (e.g., a deeper CNN, IMPALA, ResNet, or a
    #  pretrained ViT), you can swap it in via the `policy_kwargs` argument:
    #
    #      from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    #      import torch.nn as nn
    #
    #      class MyCustomEncoder(BaseFeaturesExtractor):
    #          """
    #          Custom visual encoder that replaces NatureCNN.
    #          Must accept observation_space in __init__ and expose
    #          self._features_dim (int) indicating the output feature size.
    #          """
    #          def __init__(self, observation_space, features_dim=512):
    #              super().__init__(observation_space, features_dim)
    #              # ... define your CNN / ViT / etc layers here ...
    #
    #          def forward(self, observations):
    #              # observations: (batch, C, H, W) float32 in [0, 1]
    #              # return: (batch, features_dim) float32
    #              ...
    #
    #  Then pass it to PPO like this:
    #
    #      policy_kwargs = dict(
    #          features_extractor_class=MyCustomEncoder,
    #          features_extractor_kwargs=dict(features_dim=512),
    #      )
    #      model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, ...)
    #
    #  The rest of the training pipeline remains unchanged — SB3 handles
    #  connecting the encoder output to the policy and value heads.
    # ------------------------------------------------------------------

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

    # Clean up
    env.close()
    logger.info("Training complete.")


# ===========================================================================
#  7. Main
# ===========================================================================

if __name__ == "__main__":
    train(total_timesteps=TOTAL_TIMESTEPS)
