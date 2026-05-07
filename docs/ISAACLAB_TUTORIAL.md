# IsaacLab 完整教学文档：从 IsaacGym 到 IsaacLab

## 目录
1. [概述](#概述)
2. [IsaacLab vs IsaacGym：核心区别](#isaaclab-vs-isaacgym核心区别)
3. [IsaacLab 架构详解](#isaaclab-架构详解)
4. [TienKung-Lab 项目结构](#tienkung-lab-项目结构)
5. [从 IsaacGym 迁移指南](#从-isaacgym-迁移指南)
6. [快速上手指南](#快速上手指南)
7. [核心概念详解](#核心概念详解)
8. [实战示例](#实战示例)

---

## 概述

### 什么是 IsaacLab？

**IsaacLab** 是 NVIDIA 构建在 **Isaac Sim** 之上的统一、模块化机器人学习框架。它旨在简化机器人研究中的常见工作流（如强化学习、模仿学习、运动规划等）。

### 关键特点

- **模块化设计**：环境、机器人、传感器都是独立模块，易于定制和扩展
- **Manager-based 架构**：通过管理器（Manager）系统组织代码，提高代码复用性
- **支持两种工作流**：Manager-based（模块化）和 Direct（单脚本，类似 IsaacGym）
- **基于 Isaac Sim**：利用 PhysX 物理引擎、USD 场景描述、高质量渲染
- **开箱即用**：包含大量预构建的环境、传感器和任务

### TienKung-Lab 项目做了什么？

TienKung-Lab 是一个基于 IsaacLab 的人形机器人强化学习框架，专为 **TienKung** 人形机器人设计。它实现了：

1. **AMP（Adversarial Motion Priors）训练**：使用运动重定向和对抗训练学习自然步态
2. **周期性步态奖励**：促进稳定、高效的行走和跑步
3. **传感器集成**：支持 LiDAR、深度相机等感知传感器
4. **Sim2Sim 迁移**：支持从 Isaac Sim 到 MuJoCo 的策略迁移
5. **Sim2Real 部署**：已在真实 TienKung 机器人上验证

---

## IsaacLab vs IsaacGym：核心区别

### 1. 架构层面

| 特性 | IsaacGym | IsaacLab |
|------|----------|----------|
| **基础平台** | Isaac Gym (独立) | Isaac Sim (Omniverse) |
| **代码组织** | 单脚本，每个任务独立实现 | Manager-based 模块化架构 |
| **代码复用** | 低，大量重复代码 | 高，组件可复用 |
| **配置管理** | 硬编码或简单字典 | 基于 `@configclass` 的类型安全配置 |
| **场景管理** | 手动创建和管理 | `InteractiveScene` 统一管理 |
| **传感器** | 需要手动实现 | 预构建传感器系统 |
| **渲染** | 基础渲染 | 高质量渲染，支持 tiled rendering |

### 2. 代码风格对比

#### IsaacGym 风格（单脚本）
```python
# IsaacGym 中，所有逻辑都在一个类中
class HumanoidEnv:
    def __init__(self):
        # 创建 gym
        self.gym = gymapi.Gym()
        # 手动创建所有资源
        self.create_sim()
        self.create_envs()
        # 手动管理观察、奖励、终止
        self.compute_observations()
        self.compute_rewards()
        self.compute_dones()
```

#### IsaacLab 风格（Manager-based）
```python
# IsaacLab 中，使用配置类和管理器
@configclass
class WalkEnvCfg:
    scene: BaseSceneCfg = BaseSceneCfg(...)
    rewards: RewardCfg = RewardCfg(...)
    terminations: TerminationCfg = TerminationCfg(...)
    # 配置即代码，类型安全

class WalkEnv(ManagerBasedRLEnv):
    def __init__(self, cfg: WalkEnvCfg):
        # 管理器自动处理观察、奖励、终止
        # 用户只需配置，不需要手动实现
```

### 3. 关键差异总结

1. **IsaacGym**：每个任务都是独立的脚本，代码重复多
2. **IsaacLab**：基于管理器系统，组件可复用，配置驱动

---

## IsaacLab 架构详解

### 1. 核心组件层次结构

```
Isaac Sim (底层)
    ↓
IsaacLab Core (isaaclab/)
    ├── sim/          # 仿真上下文和配置
    ├── scene/        # 场景管理（InteractiveScene）
    ├── assets/       # 资产（机器人、物体）
    ├── sensors/      # 传感器系统
    ├── actuators/    # 执行器（PD、神经网络等）
    ├── managers/     # 管理器系统 ⭐
    └── envs/         # 环境基类
        ↓
IsaacLab Tasks (isaaclab_tasks/)
    └── 预构建任务
        ↓
TienKung-Lab (legged_lab/)
    ├── envs/         # TienKung 环境
    ├── assets/       # TienKung 机器人资产
    ├── mdp/          # 奖励函数
    └── sensors/      # 自定义传感器
```

### 2. Manager 系统（核心创新）

Manager 系统是 IsaacLab 的核心架构，将环境的不同方面分解为独立的管理器：

```
ManagerBasedRLEnv
    ├── Scene Manager          # 场景管理（机器人、地形、传感器）
    ├── Observation Manager    # 观察生成
    ├── Action Manager          # 动作处理
    ├── Reward Manager          # 奖励计算
    ├── Termination Manager    # 终止条件
    ├── Event Manager           # 事件处理（随机化、重置等）
    └── Recorder Manager        # 数据记录
```

#### Manager 的优势

1. **模块化**：每个管理器负责单一职责
2. **可复用**：管理器可在不同任务间复用
3. **易测试**：可以单独测试每个管理器
4. **易扩展**：添加新功能只需添加新的管理器或配置

### 3. 配置系统（@configclass）

IsaacLab 使用 `@configclass` 装饰器创建类型安全的配置类：

```python
from isaaclab.utils import configclass

@configclass
class RobotCfg:
    action_scale: float = 0.25
    terminate_contacts_body_names: list = []
    feet_body_names: list = []

@configclass
class WalkEnvCfg:
    robot: RobotCfg = RobotCfg()
    scene: BaseSceneCfg = BaseSceneCfg(...)
    rewards: RewardCfg = RewardCfg(...)
```

**优势**：
- 类型安全（IDE 自动补全）
- 默认值管理
- 配置验证
- 易于文档化

### 4. Scene 系统（InteractiveScene）

`InteractiveScene` 统一管理场景中的所有对象：

```python
# 创建场景
scene_cfg = SceneCfg(config=cfg.scene, ...)
self.scene = InteractiveScene(scene_cfg)

# 访问场景中的对象
self.robot: Articulation = self.scene["robot"]
self.contact_sensor: ContactSensor = self.scene.sensors["contact_sensor"]
self.height_scanner: RayCaster = self.scene.sensors["height_scanner"]
```

**特点**：
- 统一接口访问所有对象
- 自动管理生命周期
- 支持命名查找
- 类型提示支持

---

## TienKung-Lab 项目结构

### 目录结构

```
TienKung-Lab/
├── IsaacLab/                    # IsaacLab 核心框架
│   ├── source/
│   │   ├── isaaclab/            # 核心库
│   │   ├── isaaclab_tasks/      # 预构建任务
│   │   ├── isaaclab_rl/         # RL 相关工具
│   │   └── isaaclab_assets/     # 预构建资产
│   └── scripts/                  # 示例脚本
│
├── legged_lab/                   # TienKung-Lab 核心代码 ⭐
│   ├── envs/                     # 环境定义
│   │   ├── base/                 # 基础环境类
│   │   │   ├── base_env.py       # 基础环境实现
│   │   │   └── base_config.py    # 基础配置
│   │   └── tienkung/             # TienKung 特定环境
│   │       ├── tienkung_env.py   # 环境实现
│   │       ├── walk_cfg.py       # 行走任务配置
│   │       └── run_cfg.py        # 跑步任务配置
│   │
│   ├── assets/                   # 机器人资产
│   │   └── tienkung2_lite/       # TienKung 机器人
│   │       ├── urdf/             # URDF 文件
│   │       ├── usd/              # USD 场景文件
│   │       └── meshes/           # 网格文件
│   │
│   ├── mdp/                      # MDP 相关（奖励函数）
│   │   └── rewards.py           # 奖励函数实现
│   │
│   ├── sensors/                  # 自定义传感器
│   │   ├── camera/               # 相机传感器
│   │   └── lidar/                # LiDAR 传感器
│   │
│   ├── terrains/                  # 地形生成
│   │   └── terrain_generator_cfg.py
│   │
│   └── scripts/                  # 脚本
│       ├── train.py              # 训练脚本
│       ├── play.py               # 播放脚本
│       └── sim2sim.py            # Sim2Sim 迁移
│
├── rsl_rl/                       # RSL-RL 算法库
│   └── rsl_rl/
│       ├── algorithms/           # 算法（PPO、AMPPPO）
│       ├── runners/               # Runner（OnPolicyRunner、AmpOnPolicyRunner）
│       └── modules/              # 网络模块
│
└── docs/                         # 文档
```

### 关键文件说明

#### 1. `legged_lab/envs/base/base_env.py`
- **作用**：所有环境的基类
- **继承**：`VecEnv`（来自 rsl_rl）
- **功能**：
  - 初始化仿真上下文
  - 创建场景
  - 管理观察、动作、奖励缓冲区
  - 实现 `step()` 和 `reset()` 接口

#### 2. `legged_lab/envs/tienkung/tienkung_env.py`
- **作用**：TienKung 环境的具体实现
- **继承**：`BaseEnv`
- **功能**：
  - 实现 AMP 观察
  - 实现奖励计算
  - 实现终止条件
  - 支持传感器集成

#### 3. `legged_lab/envs/tienkung/walk_cfg.py`
- **作用**：行走任务的配置
- **内容**：
  - 场景配置（机器人、地形）
  - 奖励配置
  - 命令配置（速度命令）
  - 域随机化配置

#### 4. `legged_lab/scripts/train.py`
- **作用**：训练入口脚本
- **流程**：
  1. 解析命令行参数
  2. 启动 Isaac Sim 应用
  3. 创建环境
  4. 创建 Runner（OnPolicyRunner 或 AmpOnPolicyRunner）
  5. 开始训练

---

## 从 IsaacGym 迁移指南

### 1. 概念映射

| IsaacGym | IsaacLab | 说明 |
|----------|----------|------|
| `gymapi.Gym()` | `SimulationContext` | 仿真上下文 |
| `gym.create_sim()` | `InteractiveScene` | 场景创建 |
| `gym.create_env()` | `scene.add()` | 添加对象到场景 |
| 手动计算观察 | `ObservationManager` | 观察管理器 |
| 手动计算奖励 | `RewardManager` | 奖励管理器 |
| 手动重置 | `EventManager` | 事件管理器 |
| 硬编码配置 | `@configclass` | 配置类 |

### 2. 代码迁移示例

#### IsaacGym 代码
```python
class HumanoidEnv:
    def __init__(self):
        self.gym = gymapi.Gym()
        self.sim = self.gym.create_sim(...)
        self.create_envs()
    
    def create_envs(self):
        # 手动创建每个环境
        for i in range(self.num_envs):
            env = self.gym.create_env(...)
            # 手动添加机器人
            actor = self.gym.create_actor(...)
    
    def compute_observations(self):
        # 手动计算所有观察
        obs = []
        for i in range(self.num_envs):
            # 手动获取状态
            root_states = self.gym.get_actor_root_state_tensor(...)
            # 手动拼接观察
            obs.append([...])
        return torch.stack(obs)
```

#### IsaacLab 代码
```python
@configclass
class HumanoidEnvCfg:
    scene: BaseSceneCfg = BaseSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
        robot=HumanoidCfg(...)
    )
    rewards: RewardCfg = RewardCfg(...)

class HumanoidEnv(ManagerBasedRLEnv):
    def __init__(self, cfg: HumanoidEnvCfg):
        # 场景自动创建
        # 观察、奖励自动计算
        # 只需配置，无需手动实现
```

### 3. 迁移步骤

1. **理解配置系统**
   - 将硬编码参数移到 `@configclass` 配置类
   - 使用类型提示

2. **使用 Scene 系统**
   - 用 `InteractiveScene` 替换手动创建环境
   - 用 `scene["robot"]` 访问对象

3. **使用 Manager 系统**
   - 用 `RewardManager` 替换手动奖励计算
   - 用 `ObservationManager` 替换手动观察计算
   - 用 `EventManager` 替换手动重置逻辑

4. **使用预构建组件**
   - 使用预构建的传感器（ContactSensor、RayCaster）
   - 使用预构建的执行器（PD、神经网络）

---

## 快速上手指南

### 1. 理解训练流程

```python
# 1. 启动 Isaac Sim 应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. 获取环境配置和类
env_cfg, agent_cfg = task_registry.get_cfgs("walk")
env_class = task_registry.get_task_class("walk")

# 3. 创建环境
env = env_class(env_cfg, headless=True)

# 4. 创建 Runner
runner = AmpOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir)

# 5. 开始训练
runner.learn(num_learning_iterations=50000)
```

### 2. 理解环境接口

```python
# VecEnv 接口（来自 rsl_rl）
class VecEnv:
    def step(self, actions) -> Tuple[obs, rewards, dones, infos]
    def reset(self, env_ids=None) -> Tuple[obs, infos]
    def get_observations(self) -> Tuple[obs, extras]
    @property
    def num_envs(self) -> int
    @property
    def num_actions(self) -> int
```

### 3. 理解配置层次

```
WalkEnvCfg (任务配置)
    ├── scene: BaseSceneCfg (场景配置)
    │   ├── robot: ArticulationCfg (机器人配置)
    │   ├── terrain_generator: TerrainGeneratorCfg (地形配置)
    │   └── sensors: SensorCfg (传感器配置)
    │
    ├── rewards: RewardCfg (奖励配置)
    │   └── track_lin_vel_xy_exp: RewTerm (奖励项)
    │
    ├── commands: CommandsCfg (命令配置)
    │   └── ranges: CommandRangesCfg (命令范围)
    │
    └── domain_rand: DomainRandCfg (域随机化配置)
        └── events: EventCfg (事件配置)
```

### 4. 关键代码阅读顺序

1. **`legged_lab/scripts/train.py`** - 理解训练入口
2. **`legged_lab/envs/tienkung/walk_cfg.py`** - 理解配置结构
3. **`legged_lab/envs/base/base_env.py`** - 理解基础环境实现
4. **`legged_lab/envs/tienkung/tienkung_env.py`** - 理解具体环境实现
5. **`legged_lab/mdp/rewards.py`** - 理解奖励函数

---

## 核心概念详解

### 1. SimulationContext（仿真上下文）

```python
from isaaclab.sim import SimulationContext, SimulationCfg, PhysxCfg

sim_cfg = SimulationCfg(
    device="cuda:0",
    dt=0.005,  # 物理时间步
    render_interval=4,  # 渲染间隔（decimation）
    physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
)

self.sim = SimulationContext(sim_cfg)
self.sim.reset()  # 重置仿真
```

**关键点**：
- `dt`：物理时间步（通常 0.005s）
- `render_interval`：每 N 个物理步渲染一次（decimation）
- `step_dt = dt * render_interval`：环境时间步

### 2. InteractiveScene（交互场景）

```python
from isaaclab.scene import InteractiveScene

scene_cfg = SceneCfg(config=cfg.scene, physics_dt=0.005, step_dt=0.02)
self.scene = InteractiveScene(scene_cfg)

# 访问场景中的对象
self.robot: Articulation = self.scene["robot"]
self.contact_sensor: ContactSensor = self.scene.sensors["contact_sensor"]
```

**关键点**：
- 统一管理所有场景对象
- 支持命名查找
- 自动管理生命周期

### 3. Articulation（关节机器人）

```python
from isaaclab.assets.articulation import Articulation

self.robot: Articulation = self.scene["robot"]

# 访问机器人数据
joint_pos = self.robot.data.joint_pos  # [num_envs, num_joints]
joint_vel = self.robot.data.joint_vel
root_pos = self.robot.data.root_pos_w  # [num_envs, 3]
root_quat = self.robot.data.root_quat_w  # [num_envs, 4]

# 设置动作
self.robot.set_joint_position_target(target_pos)
self.robot.set_joint_velocity_target(target_vel)
self.robot.set_joint_effort_target(target_effort)
```

### 4. RewardManager（奖励管理器）

```python
from isaaclab.managers import RewardManager, RewardTermCfg as RewTerm

@configclass
class RewardCfg:
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"std": 0.5}
    )
    energy = RewTerm(func=mdp.energy, weight=-1e-3)

self.reward_manager = RewardManager(self.cfg.reward, self)
rewards = self.reward_manager.compute()  # 自动计算所有奖励项
```

**关键点**：
- 每个奖励项是一个 `RewTerm`
- `func`：奖励函数
- `weight`：权重
- `params`：参数

### 5. EventManager（事件管理器）

```python
from isaaclab.managers import EventManager, EventTermCfg as EventTerm

@configclass
class EventCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",  # "startup" 或 "reset"
        params={
            "pose_range": {"x": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {...}
        }
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-1.0, 1.0)}}
    )

self.event_manager = EventManager(self.cfg.domain_rand.events, self)
self.event_manager.apply(mode="startup")  # 启动时应用
self.event_manager.apply(mode="reset", env_ids=env_ids)  # 重置时应用
```

**事件模式**：
- `"startup"`：环境初始化时执行一次
- `"reset"`：每次重置时执行
- `"interval"`：按时间间隔执行

### 6. AMP（Adversarial Motion Priors）

AMP 是 TienKung-Lab 的核心训练方法：

```python
# 1. 加载专家运动数据
amp_data = AMPLoader(
    device,
    time_between_frames=self.env.step_dt,
    motion_files=train_cfg["amp_motion_files"]
)

# 2. 创建判别器
discriminator = Discriminator(
    amp_data.observation_dim * 2,
    train_cfg["amp_reward_coef"],
    train_cfg["amp_discr_hidden_dims"],
    device
)

# 3. 使用 AMPPPO 算法
algorithm = AMPPPO(
    policy,
    discriminator,
    amp_data,
    ...
)
```

**AMP 流程**：
1. 从专家运动数据中采样状态对
2. 判别器区分专家和策略生成的状态对
3. 奖励 = 任务奖励 + AMP 奖励（来自判别器）
4. 策略学习生成类似专家的运动

---

## 实战示例

### 示例 1：理解训练流程

```python
# legged_lab/scripts/train.py

# 1. 解析参数
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="walk")
args = parser.parse_args()

# 2. 启动应用
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# 3. 获取配置
env_cfg, agent_cfg = task_registry.get_cfgs("walk")
env_class = task_registry.get_task_class("walk")

# 4. 创建环境
env = env_class(env_cfg, headless=True)

# 5. 创建 Runner
runner = AmpOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir)

# 6. 训练
runner.learn(num_learning_iterations=50000)
```

### 示例 2：理解环境实现

```python
# legged_lab/envs/tienkung/tienkung_env.py

class TienKungEnv(VecEnv):
    def __init__(self, cfg, headless):
        # 1. 创建仿真上下文
        self.sim = SimulationContext(sim_cfg)
        
        # 2. 创建场景
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()
        
        # 3. 访问场景对象
        self.robot = self.scene["robot"]
        self.contact_sensor = self.scene.sensors["contact_sensor"]
        
        # 4. 创建管理器
        self.reward_manager = RewardManager(self.cfg.reward, self)
        self.event_manager = EventManager(self.cfg.domain_rand.events, self)
        
        # 5. 初始化缓冲区
        self.init_buffers()
        
        # 6. 重置环境
        self.reset(env_ids)
    
    def step(self, actions):
        # 1. 应用动作
        self._apply_actions(actions)
        
        # 2. 步进仿真
        for _ in range(self.cfg.sim.decimation):
            self.sim.step(render=self.sim.render)
        
        # 3. 计算观察
        obs = self.compute_current_observations()
        
        # 4. 计算奖励
        rewards = self.reward_manager.compute()
        
        # 5. 检查终止
        dones = self._check_termination()
        
        # 6. 重置完成的环境
        env_ids = dones.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            obs[env_ids], _ = self.reset(env_ids)
        
        return obs, rewards, dones, {}
```

### 示例 3：理解配置结构

```python
# legged_lab/envs/tienkung/walk_cfg.py

@configclass
class TienKungWalkFlatEnvCfg:
    # 场景配置
    scene: BaseSceneCfg = BaseSceneCfg(
        num_envs=4096,
        robot=TIENKUNG2LITE_CFG,
        terrain_type="generator",
        terrain_generator=GRAVEL_TERRAINS_CFG
    )
    
    # 奖励配置
    reward = LiteRewardCfg()  # 包含多个 RewTerm
    
    # 命令配置
    commands: CommandsCfg = CommandsCfg(
        ranges=CommandRangesCfg(
            lin_vel_x=(-0.6, 1.0),
            ang_vel_z=(-1.57, 1.57)
        )
    )
    
    # 域随机化配置
    domain_rand: DomainRandCfg = DomainRandCfg(
        events=EventCfg(
            physics_material=EventTerm(...),
            push_robot=EventTerm(...)
        )
    )
```

---

## 总结

### 关键要点

1. **IsaacLab 是框架，不是仿真器**
   - 构建在 Isaac Sim 之上
   - 提供统一的机器人学习接口

2. **Manager 系统是核心**
   - 模块化设计
   - 高代码复用性
   - 易于扩展

3. **配置驱动开发**
   - `@configclass` 提供类型安全
   - 配置即文档

4. **TienKung-Lab 项目结构清晰**
   - `envs/`：环境定义
   - `assets/`：机器人资产
   - `mdp/`：奖励函数
   - `scripts/`：训练/测试脚本

### 学习路径建议

1. **第一步**：运行示例
   ```bash
   python legged_lab/scripts/train.py --task=walk --headless --num_envs=64
   ```

2. **第二步**：阅读配置
   - 理解 `walk_cfg.py` 的结构
   - 修改配置参数，观察效果

3. **第三步**：理解环境实现
   - 阅读 `base_env.py`
   - 阅读 `tienkung_env.py`

4. **第四步**：理解奖励函数
   - 阅读 `mdp/rewards.py`
   - 尝试添加新的奖励项

5. **第五步**：理解训练流程
   - 阅读 `scripts/train.py`
   - 理解 Runner 的作用

### 常见问题

**Q: IsaacLab 和 IsaacGym 可以同时使用吗？**
A: 不需要。IsaacLab 是 IsaacGym 的替代品，功能更强大。

**Q: 如何添加新的奖励项？**
A: 在 `mdp/rewards.py` 中定义函数，在配置中添加 `RewTerm`。

**Q: 如何添加新的传感器？**
A: 继承 `SensorBase`，在场景配置中注册。

**Q: 如何修改机器人？**
A: 修改 `assets/tienkung2_lite/` 中的 URDF/USD 文件，或创建新的资产配置。

---

## 参考资料

- [IsaacLab 官方文档](https://isaac-sim.github.io/IsaacLab/)
- [Isaac Sim 文档](https://docs.omniverse.nvidia.com/isaacsim/latest/)
- [RSL-RL 文档](https://github.com/leggedrobotics/rsl_rl)
- [TienKung-Lab README](./README.md)

---

**祝学习愉快！如有问题，欢迎查阅文档或提交 Issue。**

