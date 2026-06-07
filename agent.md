# agent

## 项目结构
该项目用于一系列姿态生成（human motion）任务，其文件目录结构如下

motion_workspace/
├── repos/                  # 所有代码仓库
│   ├── motion-toy/          # 自己写的小玩具：toy predictor / toy DD
│   ├── motion-dd/           # 你的正式方法仓库，未来主线
│   ├── baselines/           # 外部 baseline 代码统一放这里
│   │   ├── siMLPe/
│   │   ├── human-motion-prediction/
│   │   └── ...
│
├── tools/                    # 各种工具代码
│   ├── data-tools/          # 数据转换、格式统一、预处理
│   ├── vis-tools/           # 可视化工具，可独立复用
│   └── README.md
│
├── datasets/                # 所有数据，不进 git
│   ├── raw/
│   ├── processed/
│   ├── toy/
│   └── README.md
│
├── outputs/                 # 所有实验结果
│   ├── toy/
│   ├── baselines/
│   ├── motion-dd/
│   └── figures/
│
├── envs/                    # 环境文件/依赖记录
│   ├── motiondd.yaml
│   ├── simlpe.yaml
│   └── README.md
│
└── scripts/                 # 跨 repo 的服务器脚本
    ├── sync_code.sh
    ├── check_gpu.sh
    └── start.sh

## 项目要求：
* 在本机上，不需要安装任何环境，也无需执行指令（除了极少数的我允许的内容），因为代码的执行环境在当前机器并不存在。
* 代码保证可读性，使用简单英语+中文进行注释
* 保证代码低耦合度，通用功能尽可能模块化，函数化

## 注意事项：持续更新中

在tools和datasets的readme.md中，均需要注明，被转化的数据格式为npz，其中包括
(注：不需要全部包括，但至少应该遵循如下范式，该内容为参考内容，具体转化格式由脚本决定)
```
motions      # [N, T, J, 3]
joint_names  # [J]
edges        # [E, 2]
fps          # int

```

