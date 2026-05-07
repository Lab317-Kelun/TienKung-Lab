# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """
    随机化刚体质心(CoM)位置,通过在给定范围内添加随机值。
    
    用途:
        模拟制造误差、装配误差导致的质心偏移,增强训练鲁棒性。
    
    Args:
        env: 环境实例
        env_ids: 要随机化的环境ID,None表示所有环境
        com_range: 质心偏移范围,例如 {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)}
        asset_cfg: 资产配置,指定要随机化的body
    
    Note:
        此函数使用CPU张量来设置质心,建议仅在环境初始化时使用。
    
    Example:
        ```python
        base_com=EventTerm(
            func=mdp.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
                "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
            },
        )
        ```
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()

    # Randomize the com in range
    coms[:, body_ids, :3] += rand_samples

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    随机化关节默认位置,模拟校准误差。
    
    用途:
        模拟机器人关节零点校准不准确,增强策略对初始姿态的鲁棒性。
        这会影响控制器的目标位置计算。
    
    Args:
        env: 环境实例
        env_ids: 要随机化的环境ID,None表示所有环境
        asset_cfg: 资产配置,指定要随机化的关节
        pos_distribution_params: 位置分布参数 (min, max)
        operation: 操作类型 - "add"(加法), "scale"(乘法), "abs"(绝对值)
        distribution: 分布类型 - "uniform", "log_uniform", "gaussian"
    
    Example:
        ```python
        randomize_joint_default_pos=EventTerm(
            func=mdp.randomize_joint_default_pos,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "pos_distribution_params": (-0.01, 0.01),  # ±0.01 rad
                "operation": "add",
                "distribution": "uniform",
            },
        )
        ```
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos

        action_manager = getattr(env, "action_manager", None)
        if action_manager is not None:
            action_term = action_manager.get_term("joint_pos")
            action_term._offset[env_ids, joint_ids] = pos
