"""
PPO agent for MineDojo that rewards placing obsidian blocks,
intended to encourage nether portal construction behavior.

Reward signal: derived from obs["voxel"]["block_name"] == "obsidian"
  - +1.0  for each new obsidian block placed in the voxel grid
  - -0.01 per step (time penalty to encourage efficiency)
  - +5.0  bonus if a valid nether portal frame shape is detected
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import minedojo
from minedojo.sim import InventoryItem

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecVideoRecorder
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = "OhGODHELPME"

BASE_CFG = dict(
    task_id="open-ended",
    image_size=(160, 256),
    initial_inventory=[
        InventoryItem(slot=0, name="obsidian", variant=None, quantity=64),
        InventoryItem(slot=1, name="flint_and_steel", variant=None, quantity=1),
    ],
    world_seed=SEED,
    use_voxel=True,
    voxel_size=dict(xmin=-4, ymin=-4, zmin=-4, xmax=4, ymax=4, zmax=4),
)

# Voxel grid shape derived from cfg: (x_range, y_range, z_range)
# xmin=-4..xmax=4  → 9,  ymin=-4..ymax=4 → 9,  zmin=-4..zmax=4 → 9
VOXEL_SHAPE = (9, 9, 9)


# ---------------------------------------------------------------------------
# Helper: detect a minimal nether-portal frame (4×5 obsidian rectangle)
# ---------------------------------------------------------------------------
# 14 positions of a valid nether portal frame (4 wide × 5 tall),
# as (x, y) offsets from the bottom-left corner of the frame.
# Corners are optional in Minecraft but included here for full reward.
_PORTAL_FRAME_OFFSETS = [
    (0, 0), (1, 0), (2, 0), (3, 0),  # bottom row
    (0, 4), (1, 4), (2, 4), (3, 4),  # top row
    (0, 1), (0, 2), (0, 3),           # left column
    (3, 1), (3, 2), (3, 3),           # right column
]
_PORTAL_FRAME_SIZE = len(_PORTAL_FRAME_OFFSETS)  # 14


def _score_portal_progress(obsidian_grid: np.ndarray):
    """
    Slide a 4×5 window over every z-slice of the voxel grid and count
    how many of the 14 frame positions are filled in the best candidate.

    Returns:
        best_score  int  — 0-14, how many frame positions are filled
        is_complete bool — True if all 14 positions are filled
    """
    best = 0
    for z in range(obsidian_grid.shape[2]):
        plane = obsidian_grid[:, :, z]
        for x in range(plane.shape[0] - 3):
            for y in range(plane.shape[1] - 4):
                score = sum(
                    1 for (dx, dy) in _PORTAL_FRAME_OFFSETS
                    if plane[x + dx, y + dy]
                )
                if score > best:
                    best = score
    return best, best == _PORTAL_FRAME_SIZE


def _has_portal_frame(obsidian_grid: np.ndarray) -> bool:
    _, complete = _score_portal_progress(obsidian_grid)
    return complete


# ---------------------------------------------------------------------------
# Gym wrapper
# ---------------------------------------------------------------------------
class ObsidianPortalEnv(gym.Env):
    """
    Wraps a MineDojo open-ended environment.

    Observation space:
        - "rgb":    (C, H, W) uint8 image from the first-person camera
        - "voxel":  (9,9,9)   float32  — 1.0 where obsidian, else 0.0

    Action space: inherited directly from MineDojo (MultiDiscrete).

    Reward:
        +1.0  per *new* obsidian voxel placed since last step
        -0.01 time penalty per step
        +5.0  bonus when a portal frame is first detected
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, cfg: dict = BASE_CFG, max_steps: int = 500):
        super().__init__()
        self._env = minedojo.make(**cfg)
        self._max_steps = max_steps

        # ---- observation space ----
        H, W = cfg["image_size"]
        self.observation_space = spaces.Dict(
            {
                "rgb": spaces.Box(low=0, high=255, shape=(3, H, W), dtype=np.uint8),
                "obsidian_mask": spaces.Box(
                    low=0.0, high=1.0, shape=VOXEL_SHAPE, dtype=np.float32
                ),
            }
        )

        # ---- action space ----
        # SB3 supports MultiDiscrete natively. MineDojo's space may have a
        # 2-D nvec array which SB3 rejects — we rebuild it as a clean 1-D
        # MultiDiscrete so SB3's type check always passes.
        self._action_nvec = np.array([3, 3, 4, 25, 25, 8, 244, 36], dtype=np.int64)
        self.action_space = spaces.MultiDiscrete(self._action_nvec)

        # ---- internal state ----
        self._prev_voxel_obsidian: int = 0
        self._prev_portal_score: int   = 0
        self._prev_misplaced: int       = 0
        self._portal_bonus_given: bool  = False
        self._total_steps: int          = 0  # global step counter for curriculum
        self._step_count: int = 0
        self._image_size: tuple = cfg["image_size"]  # (H, W)
        self._last_rgb: np.ndarray | None = None
        self._prev_health: float = 20.0
        self._current_health: float = 20.0

    # ------------------------------------------------------------------
    def _extract_obs(self, raw_obs: dict) -> dict:
        """Convert MineDojo obs dict → our slim obs dict."""
        rgb = raw_obs["rgb"]  # already (3, H, W) uint8
        self._last_rgb = rgb

        # MineDojo flattens voxel data into the top-level obs dict.
        # The block-name array is keyed as "voxels" (with an 's'), not "voxel".
        # Run smoke_test() with DEBUG=True to print all keys if this breaks.
        block_names: np.ndarray = raw_obs["voxels"]["block_name"]
        obsidian_mask = (block_names == "obsidian").astype(np.float32)

        # Cache masks so masked_action() can read them after each step.
        self._last_masks = raw_obs.get("masks", {})
        # Cache current health (0-20) for damage/death penalties.
        self._current_health = float(raw_obs.get("life_stats", {}).get("life", 20.0))

        return {"rgb": rgb, "obsidian_mask": obsidian_mask}

    # ------------------------------------------------------------------
    # Action space indices  MultiDiscrete([3, 3, 4, 25, 25, 8, 244, 36])
    _IDX_FORWARD   = 0   # 1=forward, 2=back
    _IDX_STRAFE    = 1   # 1=left, 2=right
    _IDX_JUMP      = 2   # 1=jump, 3=sprint
    _IDX_PITCH     = 3   # 0-24 camera pitch delta
    _IDX_YAW       = 4   # 0-24 camera yaw delta
    _IDX_FUNC      = 5   # 0=noop 1=use 2=drop 3=attack 4=craft 5=equip 6=place 7=destroy
    _IDX_CRAFT_ARG = 6   # item index for craft
    _IDX_SLOT_ARG  = 7   # inventory slot for equip/place/destroy

    _FUNC_NOOP    = 0
    _FUNC_DROP    = 2
    _FUNC_EQUIP   = 5
    _FUNC_PLACE   = 6
    _FUNC_DESTROY = 7

    def masked_action(self, model_action: np.ndarray) -> np.ndarray:
        """
        Enforces MineDojo action masks on a raw PPO action so every step is
        either a valid movement or a valid place action.

        Mask sources (from obs["masks"]):
          action_type  (8,)  bool -- which functional actions are valid
          place        (36,) bool -- which inventory slots are placeable

        Invalid functional actions fall back to noop.
        Invalid place slots fall back to the first valid placeable slot,
        or noop if nothing is placeable.
        Pure noop (no movement, no functional action) forces forward=1.
        """
        action = model_action.copy()
        masks  = self._last_masks

        # 1. Validate functional action
        func = int(action[self._IDX_FUNC])
        action_type_mask = masks.get("action_type", np.ones(8, dtype=bool))
        action_type_mask[self._FUNC_DROP]    = False  # never allow dropping items
        action_type_mask[self._FUNC_DESTROY] = False  # never allow destroying inventory slots
        if not action_type_mask[func]:
            func = self._FUNC_NOOP
            action[self._IDX_FUNC] = func

        # 2. Validate slot-based functional actions against their respective masks
        if func == self._FUNC_EQUIP:
            equip_mask = masks.get("equip", np.zeros(36, dtype=bool))
            slot = int(action[self._IDX_SLOT_ARG])
            if not equip_mask[slot]:
                valid_slots = np.where(equip_mask)[0]
                if len(valid_slots) > 0:
                    action[self._IDX_SLOT_ARG] = int(valid_slots[0])
                else:
                    action[self._IDX_FUNC]     = self._FUNC_NOOP
                    action[self._IDX_SLOT_ARG] = 0

        elif func == self._FUNC_PLACE:
            place_mask = masks.get("place", np.zeros(36, dtype=bool))
            slot = int(action[self._IDX_SLOT_ARG])
            if not place_mask[slot]:
                valid_slots = np.where(place_mask)[0]
                if len(valid_slots) > 0:
                    action[self._IDX_SLOT_ARG] = int(valid_slots[0])
                else:
                    # Nothing placeable -- revert to noop
                    action[self._IDX_FUNC]     = self._FUNC_NOOP
                    action[self._IDX_SLOT_ARG] = 0

        # 2b. Validate destroy slot -- destroy mask is only true for non-empty
        #     slots; attempting to destroy air raises a ValueError in MineDojo.
        elif func == self._FUNC_DESTROY:
            destroy_mask = masks.get("destroy", np.zeros(36, dtype=bool))
            slot = int(action[self._IDX_SLOT_ARG])
            if not destroy_mask[slot]:
                valid_slots = np.where(destroy_mask)[0]
                if len(valid_slots) > 0:
                    action[self._IDX_SLOT_ARG] = int(valid_slots[0])
                else:
                    # Nothing to destroy -- revert to noop
                    action[self._IDX_FUNC]     = self._FUNC_NOOP
                    action[self._IDX_SLOT_ARG] = 0

        # 3. Clear craft arg unless crafting
        if func != 4:
            action[self._IDX_CRAFT_ARG] = 0

        # 4. Guarantee at least movement or a functional action
        moving = (
            action[self._IDX_FORWARD] != 0
            or action[self._IDX_STRAFE] != 0
            or action[self._IDX_JUMP]   != 0
        )
        acting = action[self._IDX_FUNC] != self._FUNC_NOOP
        if not moving and not acting:
            action[self._IDX_FORWARD] = 1  # force forward

        return action

    def _soft_respawn(self) -> None:
        """
        Reset the world state and respawn the agent without restarting
        the Java process.

        Steps:
          1. /fill — replace all obsidian in the build area with air
          2. kill_agent() — respawn the agent
          3. teleport_agent() — return to episode start position
          4. set_inventory() — restore full obsidian stack
        """
        x, y, z = self._spawn_pos
        # Clear a generous area around spawn to wipe any placed obsidian.
        # The fill range should comfortably exceed the voxel window (±4).
        r = 16
        x0, y0, z0 = int(x) - r, int(y) - r, int(z) - r
        x1, y1, z1 = int(x) + r, int(y) + r, int(z) + r
        self._env.execute_cmd(
            f"/fill {x0} {y0} {z0} {x1} {y1} {z1} air replace obsidian"
        )
        self._env.kill_agent()
        self._env.teleport_agent(x, y, z, yaw=0, pitch=0)
        self._env.set_inventory([
            InventoryItem(slot=0, name="obsidian",        variant=None, quantity=64),
            InventoryItem(slot=1, name="flint_and_steel", variant=None, quantity=1),
        ])
        # Reset health and voxel tracking to avoid spurious reward signals.
        self._prev_health         = 20.0
        self._current_health      = 20.0
        self._prev_voxel_obsidian = 0
        self._prev_portal_score   = 0
        self._prev_misplaced       = 0

    def _compute_reward(self, obsidian_mask: np.ndarray) -> float:
        bool_mask = obsidian_mask.astype(bool)
        current_score, is_complete = _score_portal_progress(bool_mask)

        # -- Flat placement reward --
        # Small reward for placing ANY block so early exploration is positive.
        # This gets the agent off the ground before it learns frame geometry.
        total_placed = int(bool_mask.sum())
        prev_total   = self._prev_voxel_obsidian
        self._prev_voxel_obsidian = total_placed
        new_placements = max(0, total_placed - prev_total)
        flat_reward = new_placements * 0.5  # +0.5 per block placed anywhere

        # -- Progress reward --
        # Additional reward specifically for blocks on valid frame positions.
        progress_delta = current_score - self._prev_portal_score
        self._prev_portal_score = current_score
        progress_reward = max(0, progress_delta) * 2.0  # +2.0 per frame position filled

        # -- Misplaced block penalty (curriculum) --
        # Start at 0 penalty so the agent freely explores placement.
        # Linearly ramp up to full -1.0 penalty over CURRICULUM_STEPS steps.
        # This gives the agent time to discover placing is good before
        # punishing it for placing in the wrong location.
        CURRICULUM_STEPS = 200_000
        misplaced_weight = min(1.0, self._total_steps / CURRICULUM_STEPS)
        misplaced = max(0, total_placed - current_score)
        misplaced_delta = misplaced - self._prev_misplaced
        self._prev_misplaced = misplaced
        misplaced_penalty = max(0, misplaced_delta) * misplaced_weight  # ramps 0→1.0

        # -- Damage penalty --
        damage_taken = self._prev_health - self._current_health
        self._prev_health = self._current_health
        damage_penalty = max(0.0, damage_taken) * 0.5
        death_penalty  = 5.0 if self._current_health <= 0.0 else 0.0

        reward = (
            flat_reward          # +0.5 per block placed anywhere
            + progress_reward    # +2.0 per frame-position filled
            - misplaced_penalty  # 0→-1.0 per wasted block (curriculum)
            - damage_penalty     # -0.5 per heart lost
        )

        # One-time portal completion bonus
        if not self._portal_bonus_given and is_complete:
            reward += 20.0
            self._portal_bonus_given = True

        return reward

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        raw_obs = self._env.reset()
        # Prevent obsidian scattering on death; we handle respawn manually.
        self._env.execute_cmd("/gamerule keepInventory true")
        # Record spawn position to teleport back to after death.
        pos = raw_obs.get("location_stats", {})
        self._spawn_pos = (
            float(pos.get("xpos", 0)),
            float(pos.get("ypos", 64)),
            float(pos.get("zpos", 0)),
        )
        self._prev_voxel_obsidian = 0
        self._prev_portal_score   = 0
        self._prev_misplaced       = 0
        self._portal_bonus_given   = False
        self._step_count = 0
        self._prev_health    = 20.0
        self._current_health = 20.0
        self._last_masks = raw_obs.get("masks", {})
        obs = self._extract_obs(raw_obs)
        return obs, {}

    def step(self, action):
        action = self.masked_action(np.asarray(action, dtype=np.int64))
        raw_obs, _inner_reward, terminated, info = self._env.step(action)
        obs = self._extract_obs(raw_obs)
        reward = self._compute_reward(obs["obsidian_mask"])
        self._step_count  += 1
        self._total_steps += 1

        # Episode ends when the agent runs out of obsidian in the voxel
        # window (inventory depleted and none placed nearby), or the step
        # limit is reached — whichever comes first.
        inv = raw_obs.get("inventory", {})
        inv_names = inv.get("name", np.array([]))
        inv_obsidian = int(np.sum(inv_names == "obsidian"))

        died            = (self._current_health <= 0.0)
        out_of_obsidian = (inv_obsidian == 0)
        time_limit_hit  = (self._step_count >= self._max_steps)

        truncated  = time_limit_hit
        terminated = terminated or out_of_obsidian  # death no longer ends episode

        if died:
            # Respawn in-place; world state (placed blocks) is preserved.
            info["reset_reason"] = "death"
            self._soft_respawn()
        elif out_of_obsidian:
            info["reset_reason"] = "inventory_depleted"
        elif time_limit_hit:
            info["reset_reason"] = "time_limit"

        return obs, reward, terminated, truncated, info

    def render(self):
        # MineDojo's render() may return a dict or a (C,H,W) array.
        # DummyVecEnv.render() expects (H,W,C) uint8 from each inner env.
        # We pull the RGB frame from the last obs instead, which is always
        # available and already in (C,H,W) format — just transpose to (H,W,C).
        if self._last_rgb is not None:
            return self._last_rgb.transpose(1, 2, 0)  # (C,H,W) -> (H,W,C)
        # Fallback before first step
        raw = self._env.render()
        if isinstance(raw, np.ndarray) and raw.ndim == 3:
            if raw.shape[0] in (1, 3, 4):   # (C,H,W)
                return raw.transpose(1, 2, 0)
            return raw                        # already (H,W,C)
        return np.zeros((*self._image_size, 3), dtype=np.uint8)

    def close(self):
        self._env.close()


# ---------------------------------------------------------------------------
# Custom feature extractor (CNN for RGB + MLP for voxel, fused)
# ---------------------------------------------------------------------------
class ObsidianExtractor(BaseFeaturesExtractor):
    """
    Encodes:
      rgb   → small CNN  → 256-d vector
      voxel → flatten + MLP → 64-d vector
    Concatenates → 320-d features
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 320):
        super().__init__(observation_space, features_dim)

        H, W = observation_space["rgb"].shape[1], observation_space["rgb"].shape[2]

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute CNN output size dynamically
        dummy = torch.zeros(1, 3, H, W)
        cnn_out = self.cnn(dummy).shape[1]

        self.cnn_head = nn.Sequential(
            nn.Linear(cnn_out, 256),
            nn.ReLU(),
        )

        voxel_flat = int(np.prod(VOXEL_SHAPE))
        self.voxel_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(voxel_flat, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        rgb = observations["rgb"].float() / 255.0
        voxel = observations["obsidian_mask"]

        cnn_feat = self.cnn_head(self.cnn(rgb))
        vox_feat = self.voxel_head(voxel)

        return torch.cat([cnn_feat, vox_feat], dim=1)


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------
def make_env(max_steps: int = 500):
    def _init():
        env = ObsidianPortalEnv(cfg=BASE_CFG, max_steps=max_steps)
        return env
    return _init


def train(
    total_timesteps: int = 1_000_000,
    n_envs: int = 1,           # increase if you have multiple GPU/CPU cores
    save_path: str = "./checkpoints",
    log_path: str = "./logs",
    record_video: bool = False,
    video_path: str = "./videos",
    video_freq: int = 10_000,  # record a clip every N steps
    video_length: int = 500,   # length of each clip in steps
    max_steps: int = 500,      # max steps per episode
):
    print("Building vectorised environment …")
    vec_env = DummyVecEnv([make_env(max_steps=max_steps) for _ in range(n_envs)])
    vec_env.render_mode = "rgb_array"
    vec_env = VecMonitor(vec_env, log_path)
    vec_env.render_mode = "rgb_array"  # VecMonitor doesn't propagate render_mode

    if record_video:
        vec_env = VecVideoRecorder(
            vec_env,
            video_folder=video_path,
            record_video_trigger=lambda step: step % video_freq == 0,
            video_length=video_length,
            name_prefix="ppo_obsidian",
        )
        print(f"Video recording enabled → {video_path}  (every {video_freq:,} steps, {video_length} steps long)")

    policy_kwargs = dict(
        features_extractor_class=ObsidianExtractor,
        features_extractor_kwargs=dict(features_dim=320),
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
    )

    model = PPO(
        policy="MultiInputPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=512,           # rollout length per env
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,         # encourage exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=log_path,
        verbose=1,
        policy_kwargs=policy_kwargs,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000 // n_envs,
        save_path=save_path,
        name_prefix="ppo_obsidian",
    )

    print(f"Training PPO for {total_timesteps:,} timesteps …")
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_cb,
        progress_bar=True,
    )

    model.save(f"{save_path}/ppo_obsidian_final")
    print("Done. Model saved.")
    return model


# ---------------------------------------------------------------------------
# Quick smoke-test (single rollout, no training)
# ---------------------------------------------------------------------------
def smoke_test(debug: bool = True):
    print("Running smoke test …")
    env = ObsidianPortalEnv(cfg=BASE_CFG)

    # Print raw MineDojo obs keys before wrapping, so we can verify the
    # correct path to voxel block names.
    if debug:
        raw_env = minedojo.make(**BASE_CFG)
        raw_obs = raw_env.reset()
        print("  Raw MineDojo obs keys:", list(raw_obs.keys()))
        if "voxels" in raw_obs:
            print("  raw_obs['voxels'] keys:", list(raw_obs["voxels"].keys()))
        elif "voxel" in raw_obs:
            print("  raw_obs['voxel'] keys:", list(raw_obs["voxel"].keys()))
        else:
            print("  WARNING: neither 'voxel' nor 'voxels' found in raw obs!")
            print("  All keys:", {k: type(v) for k, v in raw_obs.items()})
        raw_env.close()

    obs, _ = env.reset()
    print(f"  rgb shape        : {obs['rgb'].shape}")
    print(f"  obsidian_mask shape : {obs['obsidian_mask'].shape}")
    print(f"  obsidian voxels in initial obs: {obs['obsidian_mask'].sum():.0f}")

    total_reward = 0.0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"  step {step:2d} | obsidian={obs['obsidian_mask'].sum():.0f} | reward={reward:.3f}")
        if terminated or truncated:
            break

    print(f"  Cumulative reward over smoke test: {total_reward:.3f}")
    env.close()
    print("Smoke test complete.")


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Record a video from a saved checkpoint
# ---------------------------------------------------------------------------
def record_checkpoint(
    model_path: str,
    video_path: str = "./videos",
    video_length: int = 500,
    max_steps: int = 500,
):
    """
    Load a saved PPO model and record a single episode to an MP4.

    Args:
        model_path:   path to a .zip checkpoint, e.g. ./checkpoints/ppo_obsidian_final
        video_path:   directory to write the MP4 into
        video_length: max frames to record
        max_steps:    episode step limit passed to the env
    """
    import os
    os.makedirs(video_path, exist_ok=True)

    print(f"Loading model from {model_path} ...")
    env = ObsidianPortalEnv(cfg=BASE_CFG, max_steps=max_steps)

    # Wrap in a VecEnv so VecVideoRecorder is happy
    vec_env = DummyVecEnv([lambda: env])
    vec_env.render_mode = "rgb_array"

    vec_env = VecVideoRecorder(
        vec_env,
        video_folder=video_path,
        record_video_trigger=lambda step: step == 0,  # start immediately
        video_length=video_length,
        name_prefix="eval",
    )

    model = PPO.load(model_path, env=vec_env)

    obs = vec_env.reset()
    total_reward = 0.0
    for step in range(video_length):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        total_reward += float(reward[0])
        reason = info[0].get("reset_reason", "")
        if done[0]:
            print(f"  Episode ended at step {step} — {reason}")
            break

    vec_env.close()
    print(f"Video saved to {video_path}/  (cumulative reward: {total_reward:.2f})")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Run a quick smoke test instead of full training")
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--save-path", default="./checkpoints")
    parser.add_argument("--log-path", default="./logs")
    parser.add_argument("--record-video", action="store_true", help="Save MP4 clips during training")
    parser.add_argument("--video-path", default="./videos")
    parser.add_argument("--video-freq", type=int, default=10_000, help="Record a clip every N steps")
    parser.add_argument("--video-length", type=int, default=500, help="Length of each clip in steps")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode before reset")
    parser.add_argument("--record", metavar="MODEL_PATH", default=None,
                        help="Load a checkpoint and record a video instead of training")
    parser.add_argument("--eval-video-path", default="./videos")
    parser.add_argument("--eval-video-length", type=int, default=500)
    args = parser.parse_args()

    if args.record:
        record_checkpoint(
            model_path=args.record,
            video_path=args.eval_video_path,
            video_length=args.eval_video_length,
            max_steps=args.max_steps,
        )
    elif args.smoke:
        smoke_test()
    else:
        train(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            save_path=args.save_path,
            log_path=args.log_path,
            record_video=args.record_video,
            video_path=args.video_path,
            video_freq=args.video_freq,
            video_length=args.video_length,
            max_steps=args.max_steps,
        )