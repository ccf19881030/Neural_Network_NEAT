# Neural_Network_NEAT 架构与代码逻辑文档

本文档用 UML 风格（Mermaid）描述项目的实现原理、代码框架与关键数据流，面向代码开发人员。

- 项目：基于 NEAT（NeuroEvolution of Augmenting Topologies）的自驱动小车
- 入口：`main.py`
- 配置：`config_file.txt`（NEAT 超参数）、`config_variables.py`（仿真/渲染常量）

> 渲染建议：VS Code 安装 `Markdown Preview Mermaid Support`，或在 GitHub / mermaid.live 中查看。

## 目录

- [1. 实现原理概述](#1-实现原理概述)
- [2. 原理框图（分层架构）](#2-原理框图分层架构)
- [3. 模块组织结构图](#3-模块组织结构图)
- [4. 类图](#4-类图)
- [5. 时序图](#5-时序图)
- [6. 流程图](#6-流程图)
- [7. 感知-决策-执行 数据流细节图](#7-感知-决策-执行-数据流细节图)
- [8. 关键调用链](#8-关键调用链)

---

## 1. 实现原理概述

该项目用 **NEAT** 训练小车在程序化生成的无限赛道上行驶。

核心闭环：

1. **NEAT 种群**（`config_file.txt`：`pop_size=50`，`tanh` 激活，`feed_forward=True`，`initial_connection=full`）维护 50 个 `Genome`。
2. 每个基因组由 `neat.nn.FeedForwardNetwork.create()` 编译成 **9 → 4 的前馈网络**。
3. 网络控制一辆 `Car`：输入 = **8 个距离传感器 + 归一化速度**；输出 = **加速 / 刹车 / 左转 / 右转**。
4. 在 `main.py` 的仿真循环中运行，按“前进距离 + 速度”奖励、按“碰撞 / 后退 / 落后 / 太慢”惩罚来累计 **fitness**。
5. 一代全部淘汰后，NEAT 做选择、交叉、变异、物种划分，产生下一代，最多 10000 代或 fitness 达 1000 停止。

关键设计点：

- **相机跟随**：`World.getScreenCoords()` 用 `世界坐标 - bestCarPos + initialPos` 实现赛道滚动效果。
- **无限道路**：`Road` 用随机控制点 + `scipy` 三次样条插值生成中心线，两侧偏移 `ROAD_WIDTH/2` 得到边界。
- **传感器**：4 条直线代表 8 个方向（每条线正反两向），与左右边界线段求交，取最近距离并归一化为 `1 - dist/SENSOR_DISTANCE`。
- **可解释性**：`NNdraw` 从 genome 的 nodes/connections 重建网络并按激活值着色（`node.py`）。

---

## 2. 原理框图（分层架构）

按“配置 → 进化 → 智能体 → 环境 → 评估/呈现”分层，箭头表示依赖或数据流方向。

```mermaid
flowchart TB
    subgraph L1["配置层"]
        direction LR
        CFG["config_file.txt<br/>NEAT 超参数"]
        CV["config_variables.py<br/>仿真常量"]
    end

    subgraph L2["进化引擎层 (neat-python)"]
        direction LR
        POP["Population<br/>种群"]
        REP["Reproduction<br/>选择/交叉/变异"]
        SPEC["SpeciesSet<br/>物种划分"]
        STAG["Stagnation<br/>停滞控制"]
    end

    subgraph L3["智能体层"]
        direction LR
        GENO["Genome<br/>基因组"]
        FFN["FeedForwardNetwork<br/>9 输入 → 4 输出"]
        CAR["Car<br/>8 传感器 + 速度"]
    end

    subgraph L4["环境层"]
        direction LR
        ROAD["Road<br/>无限赛道生成"]
        WORLD["World<br/>相机 / 最佳位置"]
    end

    subgraph L5["评估 / 呈现层"]
        direction LR
        SIM["main()<br/>仿真循环 + 适应度"]
        DRAW["NNdraw / Node<br/>网络可视化"]
        RENDER["Pygame<br/>渲染"]
    end

    CFG --> POP
    CV --> SIM
    POP --> GENO
    GENO --> FFN
    GENO --> DRAW
    FFN -->|"4 控制指令"| CAR
    ROAD --> CAR
    WORLD --> CAR
    CAR -->|"9 感知输入"| FFN
    CAR --> SIM
    SIM -->|"fitness 适应度"| POP
    SIM --> WORLD
    SIM --> DRAW
    SIM --> RENDER
    WORLD --> RENDER
    ROAD --> RENDER
    REP --> POP
    SPEC --> POP
    STAG --> POP

    classDef cfg fill:#fff3bf,stroke:#f08c00,stroke-width:2px,color:#000;
    classDef engine fill:#d0ebff,stroke:#1c7ed6,stroke-width:2px,color:#000;
    classDef agent fill:#d3f9d8,stroke:#2f9e44,stroke-width:2px,color:#000;
    classDef env fill:#ffe3e3,stroke:#e03131,stroke-width:2px,color:#000;
    classDef eval fill:#e5dbff,stroke:#7048e8,stroke-width:2px,color:#000;

    class CFG,CV cfg;
    class POP,REP,SPEC,STAG engine;
    class GENO,FFN,CAR agent;
    class ROAD,WORLD env;
    class SIM,DRAW,RENDER eval;
```

---

## 3. 模块组织结构图

文件之间的引用/导入关系（`-->` 表示 import 或直接调用，虚线表示读取配置）。

```mermaid
flowchart TB
    main["main.py<br/>入口 / 评估循环"]

    subgraph CFG["配置"]
        direction LR
        cfgvar["config_variables.py"]
        cfgfile["config_file.txt"]
    end

    subgraph DOMAIN["仿真领域"]
        direction LR
        car["car.py"]
        road["road.py"]
        world["world.py"]
        vect["vect2d.py"]
    end

    subgraph VIEW["可视化"]
        direction LR
        nndraw["NNdraw.py"]
        node["node.py"]
    end

    main --> cfgvar
    main -. 读取 .-> cfgfile
    main --> car
    main --> road
    main --> world
    main --> nndraw

    car --> cfgvar
    car --> road
    car --> vect
    road --> cfgvar
    road --> vect

    nndraw --> node
    nndraw --> car
    nndraw --> vect
    nndraw --> cfgvar
    node --> car
    node --> cfgvar

    classDef entry fill:#e5dbff,stroke:#7048e8,stroke-width:2px,color:#000;
    classDef domain fill:#d3f9d8,stroke:#2f9e44,stroke-width:2px,color:#000;
    classDef view fill:#d0ebff,stroke:#1c7ed6,stroke-width:2px,color:#000;
    classDef cfg fill:#fff3bf,stroke:#f08c00,stroke-width:2px,color:#000;

    class main entry;
    class car,road,world,vect domain;
    class nndraw,node view;
    class cfgvar,cfgfile cfg;
```

模块职责：

| 文件 | 职责 |
|------|------|
| `main.py` | 入口；`run()` 组装 NEAT，`main()` 每代评估（创建网络/车、仿真、评分、绘制） |
| `car.py` | 小车物理（`move`）、传感器（`getInputs`）、碰撞（`detectCollision`）、指令解码 `decodeCommand` |
| `road.py` | 随机控制点 + 三次样条生成无限道路与左右边界 |
| `world.py` | 相机坐标变换、最佳车位置、当前最优分数 |
| `vect2d.py` | 二维向量数据结构 |
| `node.py` / `NNdraw.py` | 从基因组重建神经网络并可视化（激活值着色） |
| `config_variables.py` / `config_file.txt` | 仿真常量 / NEAT 超参数 |

---

## 4. 类图

### 4.1 核心领域类图

```mermaid
classDiagram
    direction LR

    class vect2d {
        +float x
        +float y
        +float angle
        +co(x, y) void
        +getCo() tuple
    }

    class World {
        +tuple initialPos
        +tuple bestCarPos
        +Surface win
        +int win_width
        +int win_height
        +float score
        +NN bestNN
        +updateBestCarPos(pos) void
        +updateScore(s) void
        +getScore() float
        +getBestCarPos() tuple
        +getScreenCoords(x, y) tuple
    }

    class Car {
        +float x
        +float y
        +float rot
        +float vel
        +float acc
        +Surface img
        +Surface brake_img
        +int[] commands
        +initImgs() void
        +detectCollision(road) bool
        +getInputs(world, road) list
        +move(road, t) tuple
        +draw(world) void
    }

    class Road {
        +int num_ctrl_points
        +int next_point
        +int bottomPointIndex
        +int last_ctrl_point
        +vect2d[] ctrl_points
        +vect2d[] centerPoints
        +vect2d[] pointsLeft
        +vect2d[] pointsRight
        +calcBorders(i) void
        +createSegment(index) void
        +update(world) void
        +draw(world) void
    }

    class NN {
        +Genome genome
        +tuple pos
        +Node[] nodes
        +Connection[] connections
        +draw(world) void
    }

    class Node {
        +int id
        +int x
        +int y
        +int type
        +list color
        +string label
        +int index
        +draw_node(world) void
        +getNodeColors(world) list
    }

    class Connection {
        +Node input
        +Node output
        +float wt
        +drawConnection(world) void
    }

    World "1" o-- "0..1" NN : bestNN 显示
    Road "1" o-- "many" vect2d : 组成道路
    Car ..> Road : 读取边界 / 碰撞
    Car ..> vect2d : 传感器几何
    NN "1" *-- "many" Node : 组成
    NN "1" *-- "many" Connection : 组成
    Connection --> "1" Node : input
    Connection --> "1" Node : output
    Node ..> Car : decodeCommand()
```

### 4.2 与 NEAT 库的集成类图

```mermaid
classDiagram
    direction LR

    class Population {
        +Config config
        +Genome winner
        +run(eval_fn, n) Genome
        +add_reporter(reporter) void
    }

    class Config {
        +GenomeConfig genome_config
    }

    class DefaultGenome {
        +dict nodes
        +dict connections
        +float fitness
    }

    class FeedForwardNetwork {
        +activate(inputs) list
        +create(genome, config) FeedForwardNetwork
        +weights / biases
    }

    class StdOutReporter
    class StatisticsReporter

    Population --> Config : 使用
    Population "1" o-- "many" DefaultGenome : 种群个体
    Population --> StdOutReporter : 日志
    Population --> StatisticsReporter : 统计
    DefaultGenome ..> FeedForwardNetwork : create()
    FeedForwardNetwork ..> DefaultGenome : 由基因组构建
```

> 说明：常量 `ACC=0, BRAKE=1, TURN_LEFT=2, TURN_RIGHT=3`（指令枚举）与 `INPUT/MIDDLE/OUTPUT`（节点类型枚举）定义于 `config_variables.py:68-75`。

---

## 5. 时序图

用 `loop` / `alt` 表示嵌套循环与分支，覆盖“创建 → 仿真 → 淘汰/评分 → 繁殖”全过程。

```mermaid
sequenceDiagram
    autonumber
    actor User as 用户
    participant P as Population
    participant M as main(genomes, config)
    participant F as FeedForwardNetwork
    participant C as Car
    participant R as Road
    participant W as World

    User->>P: run(main, 10000)
    loop 每一代
        P->>M: 传入本代 genomes 进行评估
        M->>W: World(STARTING_POS, 宽, 高)
        M->>R: Road(world)
        loop 每个 genome g
            M->>F: create(g, config)
            M->>C: Car(0, 0, 0)
        end
        loop 每一帧（直到所有车被淘汰）
            M->>W: updateScore(0)
            loop 每辆存活的车 i
                M->>C: getInputs(world, road)
                C-->>M: 8 个归一化传感器
                M->>F: activate(传感器 + 归一化速度)
                F-->>M: 4 个控制指令 commands
                M->>C: move(road, t)
                C-->>M: 新坐标 (x, y)
                alt 撞车 / 落后 200 / 后退 / 太慢
                    M->>M: 移除该车，fitness 减 1
                else 正常行驶
                    M->>M: 累计 fitness 奖励
                    M->>W: 刷新 score / bestNN / bestInputs
                end
            end
            M->>W: updateBestCarPos(最前车)
            M->>R: update(world) 按需生成新路段
            M->>W: 绘制赛道 / 车辆 / 最佳网络
        end
        M-->>P: 返回本代各 genome 的 fitness
    end
    P->>P: 选择 / 交叉 / 变异 / 物种划分 → 下一代
```

---

## 6. 流程图

### 6.1 顶层 NEAT 流程

```mermaid
flowchart TD
    A["run(config_path)"] --> B["加载 config_file.txt"]
    B --> C["创建 Config / Population / Reporters"]
    C --> D["Population.run(main, 10000)"]
    D --> E["调用 main 评估每个 Genome 的 fitness"]
    E --> F{"fitness 达到阈值 1000<br/>或 达到最大代数？"}
    F -- 否 --> G["选择 / 交叉 / 变异<br/>物种划分 / 精英保留"]
    G --> E
    F -- 是 --> H["返回 winner 并结束"]

    classDef start fill:#e5dbff,stroke:#7048e8,stroke-width:2px,color:#000;
    classDef decide fill:#fff3bf,stroke:#f08c00,stroke-width:2px,color:#000;
    classDef end_ fill:#d3f9d8,stroke:#2f9e44,stroke-width:2px,color:#000;
    class A start;
    class F decide;
    class H end_;
```

### 6.2 一代仿真循环（`main`）

```mermaid
flowchart TD
    S["main(): GEN 自增"] --> A["为每个 genome 创建<br/>网络 / 车 / 可视化"]
    A --> B["创建 World 与 Road，t = 0"]
    B --> C["进入帧循环"]
    C --> D["t 自增；score 归零；处理退出事件"]
    D --> E["i = 0"]
    E --> F{"i 小于 车辆数？"}
    F -- 是 --> H["感知：8 传感器 + 归一化速度"]
    H --> I["决策：activate → 4 控制指令"]
    I --> J["执行：move → 新坐标"]
    J --> K{"t 大于 10 且<br/>撞车 / 落后 200 / 后退 / 速度太慢？"}
    K -- 是 --> L["fitness 减 1；移除该车"]
    K -- 否 --> M["累计 fitness 奖励<br/>必要时更新最优网络"]
    L --> N["i 自增"]
    M --> N
    N --> F
    F -- 否 --> P{"存活车辆数为 0？"}
    P -- 是 --> Z["结束本代，返回 NEAT"]
    P -- 否 --> Q["更新相机；road.update()；绘制一帧"]
    Q --> C

    classDef start fill:#e5dbff,stroke:#7048e8,stroke-width:2px,color:#000;
    classDef decide fill:#fff3bf,stroke:#f08c00,stroke-width:2px,color:#000;
    classDef kill fill:#ffe3e3,stroke:#e03131,stroke-width:2px,color:#000;
    classDef reward fill:#d3f9d8,stroke:#2f9e44,stroke-width:2px,color:#000;
    class S start;
    class F,K,P decide;
    class L kill;
    class M reward;
    class Z kill;
```

---

## 7. 感知-决策-执行 数据流细节图

细化单辆车在一个仿真帧内的完整数据流：**环境 → ①感知 → ②决策 → ③执行 → ④反馈**。

```mermaid
flowchart LR
    ENV["环境<br/>Road 边界点 / World 相机"]

    subgraph PER["① 感知 Perception (Car.getInputs)"]
        direction TB
        P1["8 个传感器初始化为最大距离"]
        P2["getSensorEquations()<br/>生成 4 条射线（正反 8 向）"]
        P3["遍历道路左右边界点"]
        P4["getDistance()<br/>射线与边界求交，取最近距离"]
        P5["归一化：1 - 距离 / 最大距离"]
        P1 --> P2 --> P3 --> P4 --> P5
    end

    subgraph DEC["② 决策 Decision"]
        direction TB
        D1["输入向量 9 维<br/>8 传感器 + 速度"]
        D2["FeedForwardNetwork.activate()<br/>tanh 前向传播"]
        D3["输出 4 维<br/>加速 / 刹车 / 左转 / 右转"]
        D4["decodeCommand()<br/>阈值 0.5 + 互斥判定"]
        D1 --> D2 --> D3 --> D4
    end

    subgraph EXE["③ 执行 Execution (Car.move)"]
        direction TB
        E1["应用指令更新 acc / rot"]
        E2["vel 累加并按 max_vel 裁剪"]
        E3["按朝向更新坐标 x, y"]
        E1 --> E2 --> E3
    end

    subgraph FB["④ 反馈 Feedback (main 循环)"]
        direction TB
        F1["detectCollision() 车身掩码检测"]
        F2{"撞车 / 落后 / 后退 / 太慢？"}
        F3["fitness 减 1 并淘汰该车"]
        F4["累计 fitness 奖励"]
        F5["刷新 score / bestNN / bestInputs"]
        F1 --> F2
        F2 -- 是 --> F3
        F2 -- 否 --> F4
        F4 --> F5
    end

    ENV --> P3
    P5 --> D1
    D4 --> E1
    E3 --> F1
    F5 -. 引导下一代 / 相机跟随 .-> ENV

    classDef env fill:#ffe3e3,stroke:#e03131,stroke-width:2px,color:#000;
    classDef per fill:#d0ebff,stroke:#1c7ed6,stroke-width:2px,color:#000;
    classDef dec fill:#e5dbff,stroke:#7048e8,stroke-width:2px,color:#000;
    classDef exe fill:#d3f9d8,stroke:#2f9e44,stroke-width:2px,color:#000;
    classDef fb fill:#fff3bf,stroke:#f08c00,stroke-width:2px,color:#000;

    class ENV env;
    class P1,P2,P3,P4,P5 per;
    class D1,D2,D3,D4 dec;
    class E1,E2,E3 exe;
    class F1,F2,F3,F4,F5 fb;
```

### 数据流字段对照

| 阶段 | 数据 | 来源 → 去向 | 维度 / 取值 |
|------|------|-------------|-------------|
| 感知 | `sensors[0..7]` | `Car.getInputs` → `main` | 8，[0,1]，1 = 最近 |
| 感知 | 速度 | `car.vel / MAX_VEL` | 1，[0,1] |
| 决策 | `input` | `main` → `net.activate` | 9 |
| 决策 | `car.commands` | `net.activate` → `Car` | 4，tanh 输出 |
| 执行 | `(x, y)` | `Car.move` → `main` | 世界坐标 |
| 反馈 | `ge[i].fitness` | `main` → NEAT `Genome` | 标量累计 |
| 反馈 | `world.bestNN` | `main` → `World` / `NNdraw` | 最优网络快照 |

### 单帧调用的函数时序

```mermaid
sequenceDiagram
    autonumber
    participant M as main 帧循环
    participant C as Car
    participant R as Road
    participant F as FeedForwardNetwork

    M->>C: getInputs(world, road)
    C->>C: getSensorEquations() 生成 4 条射线
    loop 遍历左右边界点
        C->>R: 读取相邻点 v[i], v[next]
        C->>C: getDistance() 求交并取最近
    end
    C-->>M: 返回 8 个归一化传感器
    M->>F: activate([传感器..., 速度])
    F-->>M: 返回 4 个控制指令
    M->>C: move(road, t)
    C->>C: decodeCommand() 互斥判定
    C->>C: 更新 acc / rot / vel / x / y
    C-->>M: 返回新坐标 (x, y)
    M->>C: detectCollision(road)
    C-->>M: 返回 True / False
    M->>M: fitness 累计 / 淘汰 / 更新最优
```

---

## 8. 关键调用链

- 评估入口：`main.py:33` `main(genomes, config)`
- 网络创建：`main.py:48` `FeedForwardNetwork.create`
- 感知：`car.py:48` `Car.getInputs` → `car.py:144` `getDistance`
- 决策：`car.py:175` `decodeCommand`
- 运动：`car.py:78` `Car.move`
- 碰撞：`car.py:33` `Car.detectCollision`
- 淘汰与评分：`main.py:82-94`
- 赛道生成：`road.py:56` `Road.createSegment` → `road.py:43` `calcBorders`
- 相机变换：`world.py:20` `World.getScreenCoords`
- 网络可视化：`NNdraw.py:11` `NN.__init__` → `node.py:15` `Node.draw_node`
