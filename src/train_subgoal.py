import os
from dataclasses import dataclass

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import minedojo
from minedojo.sim import InventoryItem

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecVideoRecorder
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import torch
import torch.nn as nn

SEED = "OhGODHELPME"
IMAGE_SIZE = (160, 256)
VOXEL_SHAPE = (9, 9, 9)
TASK_BUILD = "build"
TASK_FULL = "full"
FRAME_WIDTH = 4
FRAME_HEIGHT = 5
BOTTOM_ROW_COUNT = FRAME_WIDTH
SIDE_COLUMN_COUNT = FRAME_HEIGHT - 2
TOP_ROW_COUNT = FRAME_WIDTH
UNSAFE_IGNITE_PENALTY = 15.0
DEATH_PENALTY = 30.0

BASE_CFG = dict(
    task_id="open-ended",
    image_size=IMAGE_SIZE,
    initial_inventory=[
        InventoryItem(slot=0, name="obsidian", variant=None, quantity=64),
        InventoryItem(slot=1, name="flint_and_steel", variant=None, quantity=1),
    ],
    generate_world_type="flat",
    allow_mob_spawn=False,
    use_voxel=True,
    voxel_size=dict(xmin=-4, ymin=-4, zmin=-4, xmax=4, ymax=4, zmax=4),
)

BOTTOM_ROW_OFFSETS = {(0, 0), (1, 0), (2, 0), (3, 0)}
TOP_ROW_OFFSETS = {(0, 4), (1, 4), (2, 4), (3, 4)}
LEFT_COLUMN_OFFSETS = {(0, 1), (0, 2), (0, 3)}
RIGHT_COLUMN_OFFSETS = {(3, 1), (3, 2), (3, 3)}
PORTAL_FRAME_OFFSETS = BOTTOM_ROW_OFFSETS | TOP_ROW_OFFSETS | LEFT_COLUMN_OFFSETS | RIGHT_COLUMN_OFFSETS
INTERIOR_OFFSETS = {(1, 1), (2, 1), (1, 2), (2, 2), (1, 3), (2, 3)}

MACRO_NOOP = 0
MACRO_MOVE_FORWARD = 1
MACRO_MOVE_BACKWARD = 2
MACRO_MOVE_LEFT = 3
MACRO_MOVE_RIGHT = 4
MACRO_LOOK_LEFT = 5
MACRO_LOOK_RIGHT = 6
MACRO_LOOK_UP = 7
MACRO_LOOK_DOWN = 8
MACRO_PLACE_OBSIDIAN = 9
MACRO_JUMP_PLACE_OBSIDIAN = 10
MACRO_IGNITE_PORTAL = 11

MACRO_NAMES = [
    "noop",
    "move_forward",
    "move_backward",
    "move_left",
    "move_right",
    "look_left",
    "look_right",
    "look_up",
    "look_down",
    "place_obsidian",
    "jump_place_obsidian",
    "ignite_portal",
]


@dataclass
class PortalProgress:
    matched: int = 0
    bottom: int = 0
    left: int = 0
    right: int = 0
    top: int = 0
    interior: int = 0

    @property
    def complete(self) -> bool:
        return self.matched >= len(PORTAL_FRAME_OFFSETS)

    @property
    def bottom_complete(self) -> bool:
        return self.bottom >= BOTTOM_ROW_COUNT

    @property
    def left_complete(self) -> bool:
        return self.left >= SIDE_COLUMN_COUNT

    @property
    def right_complete(self) -> bool:
        return self.right >= SIDE_COLUMN_COUNT

    @property
    def top_complete(self) -> bool:
        return self.top >= TOP_ROW_COUNT


def _best_portal_progress(obsidian_mask: np.ndarray, portal_mask: np.ndarray) -> PortalProgress:
    best = PortalProgress()
    for z in range(obsidian_mask.shape[2]):
        plane = obsidian_mask[:, :, z]
        portal_plane = portal_mask[:, :, z]
        for x in range(plane.shape[0] - (FRAME_WIDTH - 1)):
            for y in range(plane.shape[1] - (FRAME_HEIGHT - 1)):
                bottom = sum(1 for dx, dy in BOTTOM_ROW_OFFSETS if plane[x + dx, y + dy])
                left = sum(1 for dx, dy in LEFT_COLUMN_OFFSETS if plane[x + dx, y + dy])
                right = sum(1 for dx, dy in RIGHT_COLUMN_OFFSETS if plane[x + dx, y + dy])
                top = sum(1 for dx, dy in TOP_ROW_OFFSETS if plane[x + dx, y + dy])
                matched = bottom + left + right + top
                interior = sum(1 for dx, dy in INTERIOR_OFFSETS if portal_plane[x + dx, y + dy])
                candidate = PortalProgress(
                    matched=matched,
                    bottom=bottom,
                    left=left,
                    right=right,
                    top=top,
                    interior=interior,
                )
                if (candidate.matched, candidate.interior, candidate.bottom, candidate.left + candidate.right + candidate.top) > (
                    best.matched,
                    best.interior,
                    best.bottom,
                    best.left + best.right + best.top,
                ):
                    best = candidate
    return best


class PortalExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 352):
        super().__init__(observation_space, features_dim)
        h, w = observation_space["rgb"].shape[1], observation_space["rgb"].shape[2]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        dummy = torch.zeros(1, 3, h, w)
        cnn_out = self.cnn(dummy).shape[1]
        self.rgb_head = nn.Sequential(nn.Linear(cnn_out, 256), nn.ReLU())
        voxel_flat = int(np.prod(VOXEL_SHAPE))
        self.voxel_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(voxel_flat, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        state_dim = observation_space["state"].shape[0]
        self.state_head = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        rgb = observations["rgb"].float() / 255.0
        voxel = observations["obsidian_mask"]
        state = observations["state"]
        rgb_feat = self.rgb_head(self.cnn(rgb))
        voxel_feat = self.voxel_head(voxel)
        state_feat = self.state_head(state)
        return torch.cat([rgb_feat, voxel_feat, state_feat], dim=1)


class ObsidianPortalEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    _IDX_FORWARD = 0
    _IDX_STRAFE = 1
    _IDX_JUMP = 2
    _IDX_PITCH = 3
    _IDX_YAW = 4
    _IDX_FUNC = 5
    _IDX_CRAFT_ARG = 6
    _IDX_SLOT_ARG = 7

    _FUNC_NOOP = 0
    _FUNC_USE = 1
    _FUNC_EQUIP = 5
    _FUNC_PLACE = 6

    def __init__(self, cfg: dict = BASE_CFG, max_steps: int = 180, task_mode: str = TASK_FULL):
        super().__init__()
        self._cfg = dict(cfg)
        self._env = minedojo.make(**self._cfg)
        self._max_steps = max_steps
        self._task_mode = task_mode
        h, w = self._cfg["image_size"]
        self.observation_space = spaces.Dict(
            {
                "rgb": spaces.Box(low=0, high=255, shape=(3, h, w), dtype=np.uint8),
                "obsidian_mask": spaces.Box(low=0.0, high=1.0, shape=VOXEL_SHAPE, dtype=np.float32),
                "state": spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32),
            }
        )
        self.action_space = spaces.Discrete(len(MACRO_NAMES))
        self._action_nvec = np.array([3, 3, 4, 25, 25, 8, 244, 36], dtype=np.int64)
        self._center_pitch = 12
        self._center_yaw = 12
        self._step_count = 0
        self._last_rgb = None
        self._last_masks = {}
        self._current_health = 20.0
        self._prev_health = 20.0
        self._progress = PortalProgress()
        self._portal_lit = False
        self._phase = "build"

    def _base_action(
        self,
        *,
        forward: int = 0,
        strafe: int = 0,
        jump: int = 0,
        pitch: int | None = None,
        yaw: int | None = None,
        func: int = 0,
        craft: int = 0,
        slot: int = 0,
    ) -> np.ndarray:
        return np.array(
            [
                forward,
                strafe,
                jump,
                self._center_pitch if pitch is None else pitch,
                self._center_yaw if yaw is None else yaw,
                func,
                craft,
                slot,
            ],
            dtype=np.int64,
        )

    def _masked_action(self, action: np.ndarray) -> np.ndarray:
        action = action.copy()
        masks = self._last_masks
        func = int(action[self._IDX_FUNC])
        action_type_mask = masks.get("action_type", np.ones(8, dtype=bool))
        if not action_type_mask[func]:
            action[self._IDX_FUNC] = self._FUNC_NOOP
            func = self._FUNC_NOOP
        if func == self._FUNC_EQUIP:
            equip_mask = masks.get("equip", np.zeros(36, dtype=bool))
            slot = int(action[self._IDX_SLOT_ARG])
            if not equip_mask[slot]:
                valid = np.where(equip_mask)[0]
                if len(valid) > 0:
                    action[self._IDX_SLOT_ARG] = int(valid[0])
                else:
                    action[self._IDX_FUNC] = self._FUNC_NOOP
                    action[self._IDX_SLOT_ARG] = 0
        elif func == self._FUNC_PLACE:
            place_mask = masks.get("place", np.zeros(36, dtype=bool))
            slot = int(action[self._IDX_SLOT_ARG])
            if not place_mask[slot]:
                valid = np.where(place_mask)[0]
                if len(valid) > 0:
                    action[self._IDX_SLOT_ARG] = int(valid[0])
                else:
                    action[self._IDX_FUNC] = self._FUNC_NOOP
                    action[self._IDX_SLOT_ARG] = 0
        if action[self._IDX_FUNC] != 4:
            action[self._IDX_CRAFT_ARG] = 0
        return action

    def _macro_sequence(self, action: int) -> list[np.ndarray]:
        if action == MACRO_NOOP:
            return [self._base_action()]
        if action == MACRO_MOVE_FORWARD:
            return [self._base_action(forward=1)] * 3 + [self._base_action()]
        if action == MACRO_MOVE_BACKWARD:
            return [self._base_action(forward=2)] * 3 + [self._base_action()]
        if action == MACRO_MOVE_LEFT:
            return [self._base_action(strafe=1)] * 3 + [self._base_action()]
        if action == MACRO_MOVE_RIGHT:
            return [self._base_action(strafe=2)] * 3 + [self._base_action()]
        if action == MACRO_LOOK_LEFT:
            return [self._base_action(yaw=self._center_yaw - 1), self._base_action()]
        if action == MACRO_LOOK_RIGHT:
            return [self._base_action(yaw=self._center_yaw + 1), self._base_action()]
        if action == MACRO_LOOK_UP:
            return [self._base_action(pitch=self._center_pitch - 1), self._base_action()]
        if action == MACRO_LOOK_DOWN:
            return [self._base_action(pitch=self._center_pitch + 1), self._base_action()]
        if action == MACRO_PLACE_OBSIDIAN:
            return [self._base_action(func=self._FUNC_PLACE, slot=0), self._base_action(), self._base_action()]
        if action == MACRO_JUMP_PLACE_OBSIDIAN:
            return [
                self._base_action(jump=1),
                self._base_action(jump=1),
                self._base_action(jump=1),
                self._base_action(func=self._FUNC_PLACE, slot=0),
                self._base_action(),
                self._base_action(),
            ]
        if action == MACRO_IGNITE_PORTAL:
            return [
                self._base_action(func=self._FUNC_EQUIP, slot=1),
                self._base_action(),
                self._base_action(func=self._FUNC_USE, slot=0),
                self._base_action(),
                self._base_action(),
            ]
        return [self._base_action()]

    def _update_phase(self) -> None:
        if self._task_mode == TASK_BUILD:
            self._phase = "build"
            return
        if self._portal_lit:
            self._phase = "done"
        elif self._progress.complete:
            self._phase = "light"
        else:
            self._phase = "build"

    def _state_vector(self) -> np.ndarray:
        return np.array(
            [
                self._progress.matched / len(PORTAL_FRAME_OFFSETS),
                self._progress.bottom / BOTTOM_ROW_COUNT,
                self._progress.left / SIDE_COLUMN_COUNT,
                self._progress.right / SIDE_COLUMN_COUNT,
                self._progress.top / TOP_ROW_COUNT,
                1.0 if self._phase == "build" else 0.0,
                1.0 if self._phase == "light" else 0.0,
                np.clip(self._current_health / 20.0, 0.0, 1.0),
            ],
            dtype=np.float32,
        )

    def _extract_obs(self, raw_obs: dict) -> dict:
        rgb = raw_obs["rgb"]
        self._last_rgb = rgb
        block_names = raw_obs["voxels"]["block_name"]
        obsidian_mask = block_names == "obsidian"
        portal_mask = np.isin(block_names, np.array(["portal", "nether_portal"], dtype=object))
        self._last_masks = raw_obs.get("masks", {})
        self._current_health = float(raw_obs.get("life_stats", {}).get("life", 20.0))
        self._progress = _best_portal_progress(obsidian_mask.astype(bool), portal_mask.astype(bool))
        self._portal_lit = self._progress.interior > 0
        self._update_phase()
        return {
            "rgb": rgb,
            "obsidian_mask": obsidian_mask.astype(np.float32),
            "state": self._state_vector(),
        }

    def _gate_action(self, action: int) -> tuple[int, float]:
        if action == MACRO_IGNITE_PORTAL and (self._task_mode == TASK_BUILD or self._phase == "build"):
            return MACRO_NOOP, 0.5
        if self._phase == "light" and action in {MACRO_PLACE_OBSIDIAN, MACRO_JUMP_PLACE_OBSIDIAN}:
            return MACRO_NOOP, 0.25
        return action, 0.0

    def _compute_reward(
        self,
        prev_progress: PortalProgress,
        prev_portal_lit: bool,
        inner_steps: int,
        gating_penalty: float,
        unsafe_ignite: bool,
    ) -> float:
        reward = -0.02 * inner_steps - gating_penalty
        match_delta = self._progress.matched - prev_progress.matched
        if match_delta > 0:
            reward += 2.0 * match_delta
        if self._progress.bottom_complete and not prev_progress.bottom_complete:
            reward += 4.0
        if self._progress.left_complete and not prev_progress.left_complete:
            reward += 3.0
        if self._progress.right_complete and not prev_progress.right_complete:
            reward += 3.0
        if self._progress.top_complete and not prev_progress.top_complete:
            reward += 4.0
        if self._progress.complete and not prev_progress.complete:
            reward += 18.0 if self._task_mode == TASK_BUILD else 15.0
        if self._task_mode == TASK_FULL and self._portal_lit and not prev_portal_lit:
            reward += 40.0
        damage_taken = max(0.0, self._prev_health - self._current_health)
        self._prev_health = self._current_health
        reward -= 0.5 * damage_taken
        if unsafe_ignite:
            reward -= UNSAFE_IGNITE_PENALTY
        if self._current_health <= 0.0:
            reward -= DEATH_PENALTY
        return reward

    def reset(self, *, seed=None, options=None):
        raw_obs = self._env.reset()
        self._step_count = 0
        self._last_masks = raw_obs.get("masks", {})
        self._last_rgb = raw_obs["rgb"]
        self._current_health = float(raw_obs.get("life_stats", {}).get("life", 20.0))
        self._prev_health = self._current_health
        self._portal_lit = False
        self._progress = PortalProgress()
        obs = self._extract_obs(raw_obs)
        return obs, {}

    def step(self, action):
        requested_action = int(action)
        executed_action, gating_penalty = self._gate_action(requested_action)
        prev_progress = PortalProgress(
            matched=self._progress.matched,
            bottom=self._progress.bottom,
            left=self._progress.left,
            right=self._progress.right,
            top=self._progress.top,
            interior=self._progress.interior,
        )
        prev_portal_lit = self._portal_lit
        health_before = self._current_health
        obs = None
        raw_obs = None
        terminated = False
        info = {}
        inner_steps = 0
        for raw_action in self._macro_sequence(executed_action):
            raw_action = self._masked_action(raw_action)
            raw_obs, _inner_reward, terminated, info = self._env.step(raw_action)
            obs = self._extract_obs(raw_obs)
            inner_steps += 1
            if terminated:
                break
        unsafe_ignite = executed_action == MACRO_IGNITE_PORTAL and not self._portal_lit and self._current_health < health_before
        reward = self._compute_reward(prev_progress, prev_portal_lit, inner_steps, gating_penalty, unsafe_ignite)
        self._step_count += 1
        inv = raw_obs.get("inventory", {}) if raw_obs is not None else {}
        inv_names = inv.get("name", np.array([]))
        obsidian_remaining = int(np.sum(inv_names == "obsidian")) if len(inv_names) else 0
        died = self._current_health <= 0.0
        success = self._progress.complete if self._task_mode == TASK_BUILD else self._portal_lit
        terminated = bool(terminated or success or died or unsafe_ignite or obsidian_remaining == 0)
        truncated = self._step_count >= self._max_steps
        if success:
            info["reset_reason"] = "frame_complete" if self._task_mode == TASK_BUILD else "portal_lit"
        elif died:
            info["reset_reason"] = "death"
        elif unsafe_ignite:
            info["reset_reason"] = "unsafe_ignite"
        elif obsidian_remaining == 0:
            info["reset_reason"] = "inventory_depleted"
        elif truncated:
            info["reset_reason"] = "time_limit"
        return obs, reward, terminated, truncated, info

    def render(self):
        if self._last_rgb is not None:
            return self._last_rgb.transpose(1, 2, 0)
        raw = self._env.render()
        if isinstance(raw, np.ndarray) and raw.ndim == 3:
            if raw.shape[0] in (1, 3, 4):
                return raw.transpose(1, 2, 0)
            return raw
        return np.zeros((*self._cfg["image_size"], 3), dtype=np.uint8)

    def close(self):
        self._env.close()


def make_env(max_steps: int = 180, task_mode: str = TASK_FULL):
    def _init():
        return ObsidianPortalEnv(cfg=BASE_CFG, max_steps=max_steps, task_mode=task_mode)
    return _init


def build_vec_env(
    n_envs: int,
    max_steps: int,
    task_mode: str,
    monitor_path: str,
    record_video: bool,
    video_path: str,
    video_freq: int,
    video_length: int,
    video_prefix: str,
):
    vec_env = DummyVecEnv([make_env(max_steps=max_steps, task_mode=task_mode) for _ in range(n_envs)])
    vec_env.render_mode = "rgb_array"
    vec_env = VecMonitor(vec_env, monitor_path)
    vec_env.render_mode = "rgb_array"
    if record_video:
        vec_env = VecVideoRecorder(
            vec_env,
            video_folder=video_path,
            record_video_trigger=lambda step: step % video_freq == 0,
            video_length=video_length,
            name_prefix=video_prefix,
        )
    return vec_env


def make_model(vec_env, log_path: str):
    policy_kwargs = dict(
        features_extractor_class=PortalExtractor,
        features_extractor_kwargs=dict(features_dim=352),
        net_arch=dict(pi=[256, 128], vf=[256, 128]),
    )
    return PPO(
        policy="MultiInputPolicy",
        env=vec_env,
        learning_rate=2.5e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=8,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=log_path,
        verbose=1,
        policy_kwargs=policy_kwargs,
    )


def train_stage(
    model,
    timesteps: int,
    task_mode: str,
    max_steps: int,
    n_envs: int,
    save_path: str,
    log_path: str,
    record_video: bool,
    video_path: str,
    video_freq: int,
    video_length: int,
):
    stage_name = TASK_BUILD if task_mode == TASK_BUILD else TASK_FULL
    vec_env = build_vec_env(
        n_envs=n_envs,
        max_steps=max_steps,
        task_mode=task_mode,
        monitor_path=os.path.join(log_path, f"{stage_name}_monitor.csv"),
        record_video=record_video,
        video_path=video_path,
        video_freq=video_freq,
        video_length=video_length,
        video_prefix=f"ppo_portal_{stage_name}",
    )
    if model is None:
        model = make_model(vec_env, log_path=log_path)
    else:
        model.set_env(vec_env)
    checkpoint_cb = CheckpointCallback(
        save_freq=max(10000 // max(n_envs, 1), 1),
        save_path=save_path,
        name_prefix=f"ppo_portal_{stage_name}",
    )
    print(f"Training stage '{stage_name}' for {timesteps:,} timesteps ...")
    model.learn(total_timesteps=timesteps, callback=checkpoint_cb, reset_num_timesteps=(model.num_timesteps == 0), progress_bar=True)
    model.save(os.path.join(save_path, f"ppo_portal_{stage_name}_final"))
    vec_env.close()
    return model


def train(
    total_timesteps: int = 300000,
    build_timesteps: int = 120000,
    n_envs: int = 1,
    save_path: str = "./checkpoints",
    log_path: str = "./logs",
    record_video: bool = False,
    video_path: str = "./videos",
    video_freq: int = 10000,
    video_length: int = 180,
    max_steps: int = 180,
    task_mode: str = TASK_FULL,
    curriculum: bool = True,
    auto_record_final: bool = False,
    eval_video_path: str = "./videos",
    eval_video_length: int = 180,
):
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(video_path, exist_ok=True)
    os.makedirs(eval_video_path, exist_ok=True)
    if curriculum and task_mode != TASK_FULL:
        raise ValueError("Curriculum training only supports task_mode='full'.")
    model = None
    final_task_mode = TASK_BUILD if curriculum else task_mode
    if curriculum:
        build_steps = min(build_timesteps, total_timesteps)
        if build_steps > 0:
            model = train_stage(
                model=None,
                timesteps=build_steps,
                task_mode=TASK_BUILD,
                max_steps=max_steps,
                n_envs=n_envs,
                save_path=save_path,
                log_path=log_path,
                record_video=record_video,
                video_path=video_path,
                video_freq=video_freq,
                video_length=video_length,
            )
        remaining = total_timesteps - build_steps
        if remaining > 0:
            final_task_mode = TASK_FULL
            model = train_stage(
                model=model,
                timesteps=remaining,
                task_mode=TASK_FULL,
                max_steps=max_steps,
                n_envs=n_envs,
                save_path=save_path,
                log_path=log_path,
                record_video=record_video,
                video_path=video_path,
                video_freq=video_freq,
                video_length=video_length,
            )
    else:
        model = train_stage(
            model=None,
            timesteps=total_timesteps,
            task_mode=task_mode,
            max_steps=max_steps,
            n_envs=n_envs,
            save_path=save_path,
            log_path=log_path,
            record_video=record_video,
            video_path=video_path,
            video_freq=video_freq,
            video_length=video_length,
        )
    model.save(os.path.join(save_path, "ppo_portal_final"))
    print(f"Done. Final model saved -> {os.path.join(save_path, 'ppo_portal_final.zip')}")
    if auto_record_final:
        record_checkpoint(
            model_path=os.path.join(save_path, "ppo_portal_final.zip"),
            video_path=eval_video_path,
            video_length=eval_video_length,
            max_steps=max_steps,
            task_mode=final_task_mode,
        )
    return model


def smoke_test(task_mode: str = TASK_FULL, max_steps: int = 20):
    env = ObsidianPortalEnv(cfg=BASE_CFG, max_steps=max_steps, task_mode=task_mode)
    obs, _ = env.reset()
    print(f"rgb shape: {obs['rgb'].shape}")
    print(f"obsidian mask shape: {obs['obsidian_mask'].shape}")
    total_reward = 0.0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"step {step:2d} | action={MACRO_NAMES[action]} | reward={reward:.3f}")
        if terminated or truncated:
            break
    print(f"Cumulative reward: {total_reward:.3f}")
    env.close()


def record_checkpoint(
    model_path: str,
    video_path: str = "./videos",
    video_length: int = 180,
    max_steps: int = 180,
    task_mode: str = TASK_FULL,
):
    os.makedirs(video_path, exist_ok=True)
    print(f"Loading model from {model_path} ...")
    env = ObsidianPortalEnv(cfg=BASE_CFG, max_steps=max_steps, task_mode=task_mode)
    vec_env = DummyVecEnv([lambda: env])
    vec_env.render_mode = "rgb_array"
    vec_env = VecVideoRecorder(
        vec_env,
        video_folder=video_path,
        record_video_trigger=lambda step: step == 0,
        video_length=video_length,
        name_prefix=f"eval_{task_mode}",
    )
    model = PPO.load(model_path, env=vec_env)
    obs = vec_env.reset()
    total_reward = 0.0
    for step in range(video_length):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        total_reward += float(reward[0])
        if done[0]:
            print(f"Episode ended at step {step} -> {info[0].get('reset_reason', 'unknown')}")
            break
    vec_env.close()
    print(f"Video saved to {video_path}/  (cumulative reward: {total_reward:.2f})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--timesteps", type=int, default=300000)
    parser.add_argument("--build-timesteps", type=int, default=120000)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--save-path", default="./checkpoints")
    parser.add_argument("--log-path", default="./logs")
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-path", default="./videos")
    parser.add_argument("--video-freq", type=int, default=10000)
    parser.add_argument("--video-length", type=int, default=180)
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument("--task-mode", choices=[TASK_BUILD, TASK_FULL], default=TASK_FULL)
    parser.add_argument("--no-curriculum", action="store_true")
    parser.add_argument("--record", metavar="MODEL_PATH", default=None)
    parser.add_argument("--eval-video-path", default="./videos")
    parser.add_argument("--eval-video-length", type=int, default=180)
    parser.add_argument("--auto-record-final", action="store_true")
    args = parser.parse_args()

    if args.record:
        record_checkpoint(
            model_path=args.record,
            video_path=args.eval_video_path,
            video_length=args.eval_video_length,
            max_steps=args.max_steps,
            task_mode=args.task_mode,
        )
    elif args.smoke:
        smoke_test(task_mode=args.task_mode, max_steps=args.max_steps)
    else:
        train(
            total_timesteps=args.timesteps,
            build_timesteps=args.build_timesteps,
            n_envs=args.n_envs,
            save_path=args.save_path,
            log_path=args.log_path,
            record_video=args.record_video,
            video_path=args.video_path,
            video_freq=args.video_freq,
            video_length=args.video_length,
            max_steps=args.max_steps,
            task_mode=args.task_mode,
            curriculum=not args.no_curriculum,
            auto_record_final=args.auto_record_final,
            eval_video_path=args.eval_video_path,
            eval_video_length=args.eval_video_length,
        )
