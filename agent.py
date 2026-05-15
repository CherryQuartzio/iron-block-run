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
from typing import List

import gym
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
from PIL import Image

import minerl  # noqa: F401 — registers built-in MineRL envs on import
from minerl.herobraine.env_spec import EnvSpec
from minerl.herobraine.env_specs.human_controls import HumanControlEnvSpec
from minerl.herobraine.hero import handlers
from minerl.herobraine.hero.handlers.server.world import FileWorldGenerator, DrawingDecorator
from minerl.herobraine.hero.handlers.agent.start import (
    AgentStartPlacement,
    DoneOnDeath,
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

# -- Agent spawn --
SPAWN_X = 0.5
SPAWN_Y = 65.0     # Adjust Y to match your world's ground level at (0, Z=0)
SPAWN_Z = 0.5
SPAWN_YAW = 0.0    # Facing +Z direction (toward the horse / race track start)

# -- Horse spawn (directly in front of agent) --
HORSE_X = 0
HORSE_Y = 65       # Same ground level as agent
HORSE_Z = 2        # 2 blocks ahead of the agent

# -- Observation --
OBS_WIDTH = 64
OBS_HEIGHT = 64
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===========================================================================
#  1. Environment Specification (herobraine)
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
        """Load the custom race-track world from disk."""
        return [
            FileWorldGenerator(
                filename=WORLD_DIR,
                destroy_after_use=False,  # Preserve the world between episodes
            )
        ]

    def create_server_decorators(self) -> List[Handler]:
        """
        Spawn a tamed, saddled horse directly in front of the agent.

        Uses Malmo XML DrawEntity to place the horse at HORSE_X/Y/Z.
        The horse is pre-tamed (Tame:1b) and given a saddle so the agent can
        mount it immediately.
        """
        return [
            DrawingDecorator(
                f'<DrawEntity x="{HORSE_X}" y="{HORSE_Y}" z="{HORSE_Z}" '
                f'type="minecraft:horse">'
                f'<NBTData>{{Tame:1b,SaddleItem:{{id:"minecraft:saddle",Count:1b}}}}</NBTData>'
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
        """Spawn the agent at the configured coordinates."""
        return super().create_agent_start() + [
            AgentStartPlacement(
                x=SPAWN_X,
                y=SPAWN_Y,
                z=SPAWN_Z,
                yaw=SPAWN_YAW,
            ),
            DoneOnDeath(),
        ]

    def create_observables(self) -> List[TranslationHandler]:
        """Only the first-person camera view — no inventory observation."""
        return [
            handlers.POVObservation(self.resolution),
        ]

    def create_actionables(self) -> List[TranslationHandler]:
        """Standard human-like keyboard + camera actions."""
        return super().create_actionables()

    def create_rewardables(self) -> List[TranslationHandler]:
        """Reward is computed externally in the Gym wrapper."""
        return []

    def create_agent_handlers(self) -> List[Handler]:
        return []

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

# Discrete action table — maps integer index to MineRL action dict overrides
ACTION_TABLE = [
    # 0: No-op
    {},
    # 1: Forward
    {"forward": 1},
    # 2: Forward + Left
    {"forward": 1, "left": 1},
    # 3: Forward + Right
    {"forward": 1, "right": 1},
    # 4: Forward + Jump
    {"forward": 1, "jump": 1},
    # 5: Camera — look left
    {"camera": np.array([0.0, -5.0])},
    # 6: Camera — look right
    {"camera": np.array([0.0, 5.0])},
    # 7: Camera — look up
    {"camera": np.array([-5.0, 0.0])},
    # 8: Camera — look down
    {"camera": np.array([5.0, 0.0])},
]

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

    def __init__(self):
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
        minerl_action = self._map_action(action)
        obs, _minerl_reward, done, info = self._env.step(minerl_action)

        processed = self._preprocess_obs(obs)

        reward = self.compute_reward(
            obs=processed,
            prev_obs=self._prev_obs,
            action=action,
            info=info,
        )

        self._prev_obs = processed
        return processed, reward, done, info

    def compute_reward(
        self,
        obs: np.ndarray,
        prev_obs: np.ndarray,
        action: int,
        info: dict,
    ) -> float:
        """
        Compute the reward for the current step.

        TODO: Implement this based on your race track design. Possible
        signals include:

          - **Checkpoint proximity**: Grant positive reward as the agent
            passes through checkpoints along the track. Requires knowing
            checkpoint coordinates and reading them from `info` (or adding
            an ObservationFromCurrentLocation monitor to the env spec).

          - **Speed bonus**: Reward the agent for maintaining forward
            velocity (higher reward for faster movement toward the next
            checkpoint).

          - **Track deviation penalty**: Penalise the agent for moving
            too far from the track centerline (requires position info).

          - **Falling off penalty**: Large negative reward if the agent
            dismounts the horse or falls off the track.

          - **Finish line bonus**: Large positive reward for completing
            the race.

          - **Time penalty**: Small negative reward per step to encourage
            faster completion.

        Args:
            obs:      Current preprocessed observation (H×W×3 uint8).
            prev_obs: Previous preprocessed observation.
            action:   The discrete action index that was taken.
            info:     Info dict from the MineRL environment step.

        Returns:
            A float reward value.
        """
        # --- STUB: Replace with actual reward logic ---
        return 0.0

    def render(self, mode="human"):
        """Delegate rendering to the inner MineRL environment."""
        return self._env.render(mode=mode)

    def close(self):
        """Clean up the inner MineRL environment."""
        self._env.close()


# ===========================================================================
#  3. Environment Factory
# ===========================================================================

def make_env():
    """
    Factory function that returns a callable (thunk) for creating a
    HorseRaceEnv instance.  Used by SB3's DummyVecEnv.
    """
    def _init():
        return HorseRaceEnv()
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
