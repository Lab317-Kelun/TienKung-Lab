# TienKung 机器人强化学习训练文档

## 目录
1. [概述](#概述)
2. [观测空间](#观测空间)
3. [网络架构](#网络架构)
4. [奖励函数](#奖励函数)
5. [训练配置](#训练配置)
6. [代码调用流程](#代码调用流程)

---

## 概述

TienKung 是一个双足人形机器人，使用 AMP-PPO (Adversarial Motion Priors - Proximal Policy Optimization) 算法进行强化学习训练。训练任务包括行走（walk）和跑步（run）等基本运动技能。

---

## 观测空间

### Actor 观测（策略网络输入）

Actor 观测包含以下组件，按顺序拼接：

| 组件 | 维度 | 描述 | 缩放因子 |
|------|------|------|----------|
| 角速度 (ang_vel) | 3 | 机器人根部的角速度（body frame） | 1.0 |
| 投影重力 (projected_gravity) | 3 | 重力在机器人本体坐标系中的投影 | 1.0 |
| 命令 (command) | 3 | 期望的线速度 x, y 和角速度 z | 1.0 |
| 关节位置 (joint_pos) | 20 | 相对于默认位置的关节角度偏差 | 1.0 |
| 关节速度 (joint_vel) | 20 | 相对于默认速度的关节速度偏差 | 1.0 |
| 动作 (action) | 20 | 上一步执行的动作 | 1.0 |
| 步态相位 sin | 2 | 左右脚的步态相位正弦值 | - |
| 步态相位 cos | 2 | 左右脚的步态相位余弦值 | - |
| 相位比例 (phase_ratio) | 2 | 左右脚的空中比例 | - |

**当前时刻 Actor 观测维度：75**

#### 观测历史

- **历史长度**：`actor_obs_history_length = 10`
- **处理方式**：使用 `CircularBuffer` 存储最近 10 个时间步的观测
- **最终 Actor 观测维度**：75 × 10 = **750**

#### 可选扩展（如果启用）

- **高度扫描**（height_scanner）：如果 `enable_height_scan=True`，会添加高度扫描数据
- **深度相机**（depth_camera）：如果 `enable_depth_camera=True`，会添加展平的深度图像数据

### Critic 观测（价值网络输入）

Critic 观测包含 Actor 观测的所有内容，**额外添加**：

| 组件 | 维度 | 描述 | 缩放因子 |
|------|------|------|----------|
| 根线速度 (root_lin_vel) | 3 | 机器人根部的线速度（body frame） | 1.0 |
| 脚部接触 (feet_contact) | 1 | 脚部是否接触地面（布尔值） | - |

**当前时刻 Critic 观测维度：79**

#### 观测历史

- **历史长度**：`critic_obs_history_length = 10`
- **最终 Critic 观测维度**：79 × 10 = **790**

### 观测处理流程

```
环境计算观测
    ↓
添加噪声（可选，仅对 Actor 观测）
    ↓
存储到历史缓冲区（CircularBuffer）
    ↓
展平历史观测（reshape 成 [num_envs, obs_dim * history_length]）
    ↓
裁剪到范围 [-100, 100]
    ↓
传递给 RSL-RL Runner
    ↓
观测归一化（EmpiricalNormalization，如果启用）
    ↓
直接输入 MLP 网络（无 CNN 预处理）
```

**重要说明**：
- ✅ **观测历史通过展平（flatten）方式处理**，不是通过 RNN/LSTM
- ✅ **没有 CNN 处理**，直接是 MLP（多层感知机）
- ✅ 观测归一化是可选的（`empirical_normalization=False` 时使用 Identity）

---

## 网络架构

### Actor 网络（策略网络）

**结构**：MLP（多层感知机）

```
输入层: 750 (actor_obs) → 512
隐藏层1: 512 → 256 (ELU激活)
隐藏层2: 256 → 128 (ELU激活)
输出层: 128 → 20 (动作维度，无激活)
```

**配置**：
- 隐藏层维度：`[512, 256, 128]`
- 激活函数：`ELU`
- 输出：动作均值（mean）
- 动作噪声：可学习的标准差（`init_noise_std=1.0`）

### Critic 网络（价值网络）

**结构**：MLP（多层感知机）

```
输入层: 790 (critic_obs) → 512
隐藏层1: 512 → 256 (ELU激活)
隐藏层2: 256 → 128 (ELU激活)
输出层: 128 → 1 (价值估计，无激活)
```

**配置**：
- 隐藏层维度：`[512, 256, 128]`
- 激活函数：`ELU`
- 输出：状态价值（V值）

### 网络特点

- **无 CNN**：观测是向量形式，直接输入 MLP
- **无 RNN/LSTM**：历史观测通过展平方式处理，不是循环网络
- **共享特征提取**：Actor 和 Critic 使用独立的 MLP，不共享参数

---

## 奖励函数

所有奖励项定义在 `LiteRewardCfg` 中，总奖励 = 所有奖励项的加权和。

### 运动跟踪奖励

| 奖励项 | 权重 | 函数 | 描述 |
|--------|------|------|------|
| `track_lin_vel_xy_exp` | 1.0 | `track_lin_vel_xy_yaw_frame_exp` | 跟踪期望的 xy 平面线速度（指数奖励，std=0.5） |
| `track_ang_vel_z_exp` | 1.0 | `track_ang_vel_z_world_exp` | 跟踪期望的 z 轴角速度（指数奖励，std=0.5） |

### 稳定性奖励

| 奖励项 | 权重 | 函数 | 描述 |
|--------|------|------|------|
| `lin_vel_z_l2` | -1.0 | `lin_vel_z_l2` | 惩罚垂直方向的速度（防止跳跃） |
| `ang_vel_xy_l2` | -0.05 | `ang_vel_xy_l2` | 惩罚 x, y 轴的角速度（保持稳定） |
| `body_orientation_l2` | -2.0 | `body_orientation_l2` | 惩罚身体姿态偏差（保持直立） |
| `flat_orientation_l2` | -1.0 | `flat_orientation_l2` | 惩罚身体倾斜 |

### 能量和动作平滑性

| 奖励项 | 权重 | 函数 | 描述 |
|--------|------|------|------|
| `energy` | -1e-3 | `energy` | 惩罚能量消耗（扭矩 × 速度） |
| `dof_acc_l2` | -2.5e-7 | `joint_acc_l2` | 惩罚关节加速度（平滑运动） |
| `action_rate_l2` | -0.01 | `action_rate_l2` | 惩罚动作变化率（动作平滑） |

### 接触和碰撞

| 奖励项 | 权重 | 函数 | 描述 |
|--------|------|------|------|
| `undesired_contacts` | -1.0 | `undesired_contacts` | 惩罚不期望的接触（膝盖、肩膀、肘部、骨盆） |
| `feet_slide` | -0.25 | `feet_slide` | 惩罚脚部滑动 |
| `feet_force` | -3e-3 | `body_force` | 惩罚脚部受力过大（阈值=500N，最大奖励=400） |
| `feet_too_near` | -2.0 | `feet_too_near_humanoid` | 惩罚双脚距离过近（阈值=0.2m） |
| `feet_stumble` | -2.0 | `feet_stumble` | 惩罚脚部绊倒 |

### 关节限制

| 奖励项 | 权重 | 函数 | 描述 |
|--------|------|------|------|
| `dof_pos_limits` | -2.0 | `joint_pos_limits` | 惩罚关节超出位置限制 |
| `joint_deviation_hip` | -0.15 | `joint_deviation_l1` | 惩罚髋关节和肩关节偏差（L1距离） |
| `joint_deviation_arms` | -0.2 | `joint_deviation_l1` | 惩罚手臂关节偏差（肩部滚转、偏航） |
| `joint_deviation_legs` | -0.02 | `joint_deviation_l1` | 惩罚腿部关节偏差（髋、膝、踝关节） |

### 步态奖励

| 奖励项 | 权重 | 函数 | 描述 |
|--------|------|------|------|
| `gait_feet_frc_perio` | 1.0 | `gait_feet_frc_perio` | 奖励脚部力的周期性（delta_t=0.02s） |
| `gait_feet_spd_perio` | 1.0 | `gait_feet_spd_perio` | 奖励脚部速度的周期性 |
| `gait_feet_frc_support_perio` | 0.6 | `gait_feet_frc_support_perio` | 奖励支撑期脚部力的周期性 |

### 特定关节奖励

| 奖励项 | 权重 | 函数 | 描述 |
|--------|------|------|------|
| `ankle_torque` | -0.0005 | `ankle_torque` | 惩罚踝关节扭矩 |
| `ankle_action` | -0.001 | `ankle_action` | 惩罚踝关节动作 |
| `hip_roll_action` | -1.0 | `hip_roll_action` | 惩罚髋关节滚转动作 |
| `hip_yaw_action` | -1.0 | `hip_yaw_action` | 惩罚髋关节偏航动作 |
| `feet_y_distance` | -2.0 | `feet_y_distance` | 惩罚双脚 y 方向距离 |

### 终止惩罚

| 奖励项 | 权重 | 函数 | 描述 |
|--------|------|------|------|
| `termination_penalty` | -200.0 | `is_terminated` | 环境终止时的严重惩罚 |

### 奖励计算

每个奖励项的计算公式：
```
reward_term = func(env, **params) * weight * dt
```

总奖励：
```
total_reward = sum(all_reward_terms)
```

其中 `dt` 是环境时间步长（`step_dt = decimation * physics_dt`）。

---

## 训练配置

### 环境配置

- **任务名称**：`walk`
- **环境数量**：4096
- **设备**：`cuda:0`
- **仿真时间步**：`dt = 0.005s`
- **降采样**：`decimation = 4`（每环境步执行 4 次物理步）
- **环境时间步**：`step_dt = 0.02s`
- **最大回合长度**：20.0 秒

### 算法配置

- **算法**：AMPPPO（Adversarial Motion Priors PPO）
- **训练迭代次数**：50000
- **每环境步数**：24
- **学习率**：1.0e-3
- **折扣因子**：0.99
- **GAE lambda**：0.95
- **PPO clip 参数**：0.2
- **熵系数**：0.005
- **价值损失系数**：1.0
- **学习轮数**：5
- **小批量数量**：4
- **最大梯度范数**：1.0

### AMP 配置

- **AMP 奖励系数**：0.3
- **AMP 运动文件**：`legged_lab/envs/tienkung/datasets/motion_amp_expert/walk.txt`
- **AMP 预加载转换数**：200000
- **AMP 任务奖励插值**：0.7
- **判别器隐藏层**：`[1024, 512, 256]`

### 观测归一化

- **经验归一化**：`False`（使用 Identity，不归一化）
- **观测裁剪**：`[-100.0, 100.0]`
- **动作裁剪**：`[-100.0, 100.0]`

---

## 代码调用流程

### 1. 初始化阶段

```python
# train.py
env_cfg, agent_cfg = task_registry.get_cfgs("walk")
env = TienKungEnv(env_cfg, headless)
runner = AmpOnPolicyRunner(env, agent_cfg.to_dict(), ...)
```

### 2. 训练循环

```python
# AmpOnPolicyRunner.learn()
for iteration in range(max_iterations):
    for step in range(num_steps_per_env):
        # 1. 采样动作
        actions = alg.act(obs, privileged_obs, amp_obs)
        
        # 2. 环境步进
        obs, rewards, dones, infos = env.step(actions)
        
        # 3. 观测归一化（如果启用）
        obs = obs_normalizer(obs)
        
        # 4. 处理经验
        alg.process_env_step(rewards, dones, infos)
    
    # 5. PPO 更新（当缓冲区满时）
    alg.update()
```

### 3. 环境步进详细流程

```python
# TienKungEnv.step()
def step(actions):
    # 1. 动作处理（延迟、裁剪、缩放）
    processed_actions = process_actions(actions)
    
    # 2. 物理仿真（decimation 次）
    for _ in range(decimation):
        sim.step()
        scene.update()
    
    # 3. 更新命令
    command_generator.compute()
    
    # 4. 计算奖励
    rewards = reward_manager.compute(dt)
    
    # 5. 检查重置
    reset_buf = check_reset()
    
    # 6. 重置终止环境
    reset(reset_env_ids)
    
    # 7. 计算观测
    actor_obs, critic_obs = compute_observations()
    
    return actor_obs, rewards, reset_buf, extras
```

### 4. 观测计算流程

```python
# TienKungEnv.compute_observations()
def compute_observations():
    # 1. 计算当前时刻观测
    current_actor_obs, current_critic_obs = compute_current_observations()
    
    # 2. 添加噪声（仅 Actor）
    if add_noise:
        current_actor_obs += noise
    
    # 3. 添加到历史缓冲区
    actor_obs_buffer.append(current_actor_obs)
    critic_obs_buffer.append(current_critic_obs)
    
    # 4. 展平历史观测
    actor_obs = actor_obs_buffer.buffer.reshape(num_envs, -1)
    critic_obs = critic_obs_buffer.buffer.reshape(num_envs, -1)
    
    # 5. 裁剪
    actor_obs = clip(actor_obs, -100, 100)
    critic_obs = clip(critic_obs, -100, 100)
    
    return actor_obs, critic_obs
```

### 5. 奖励计算流程

```python
# RewardManager.compute()
def compute(dt):
    reward_buf = 0.0
    for name, term_cfg in reward_terms:
        # 调用奖励函数
        value = term_cfg.func(env, **term_cfg.params)
        # 加权并乘以时间步
        value = value * term_cfg.weight * dt
        # 累加
        reward_buf += value
    return reward_buf
```

---

## 关键文件

- **环境实现**：`legged_lab/envs/tienkung/tienkung_env.py`
- **配置文件**：`legged_lab/envs/tienkung/walk_cfg.py`
- **奖励函数**：`legged_lab/mdp/rewards.py`
- **训练脚本**：`legged_lab/scripts/train.py`
- **网络架构**：`rsl_rl/rsl_rl/modules/actor_critic.py`
- **Runner**：`rsl_rl/rsl_rl/runners/amp_on_policy_runner.py`

---

## 总结

- **观测**：Actor 750维，Critic 790维（包含10步历史）
- **网络**：MLP，无 CNN，无 RNN
- **奖励**：28个奖励项，加权求和
- **训练**：AMPPPO 算法，50000 次迭代

---

*最后更新：2025年*

