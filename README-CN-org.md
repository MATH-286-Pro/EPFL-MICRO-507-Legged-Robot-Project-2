
# 四足机器人仿真环境

此仓库包含用于四足机器人仿真的环境。

## 安装

推荐使用 Python3.6 或更高版本的 virtualenv（或 conda）环境。首先使用 pip 安装 virtualenv，然后执行以下命令：

`virtualenv {quad_env 或其他自定义名称 venv_name} --python=python3`

激活 virtualenv：

`source {VENV_PATH}/bin/activate`

你的命令行提示符应该变为：

`(venv_name) user@pc:path$`

本仓库依赖于较新的 pybullet、gym、numpy、stable-baselines3、matplotlib 等包，可使用 `pip install [PACKAGE]` 命令安装。

## 代码结构

- [env](./env) 文件夹中包含四足机器人环境文件，具体请参见 gym 仿真环境文件 [quadruped_gym_env.py](./env/quadruped_gym_env.py)，机器人的相关功能在 [quadruped.py](./env/quadruped.py)，配置变量在 [configs_a1.py](./env/configs_a1.py) 中。你需要在 [quadruped_gym_env.py](./env/quadruped_gym_env.py) 中进行修改，并仔细阅读 [quadruped.py](./env/quadruped.py) 以获取机器人状态并调用函数来解决逆运动学、返回腿部雅可比矩阵等。
- [a1_description](./a1_description) 包含机器人的网格文件和 urdf。
- [utils](./utils) 文件夹中包含文件输入输出和绘图辅助工具。
- [hopf_network.py](./env/hopf_network.py) 提供了一个 CPG 类的框架用于实现各种步态，[run_cpg.py](run_cpg.py) 将这些关节指令映射到 [quadruped_gym_env](./env/quadruped_gym_env.py) 类的实例中执行。请仔细完成这些文件。
- [run_sb3.py](./run_sb3.py) 和 [load_sb3.py](./load_sb3.py) 提供了基于 [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) 的强化学习算法训练接口。请仔细阅读文档以了解不同算法和训练超参数。

## 代码资源
- [PyBullet 快速入门指南](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.2ye70wns7io3) 是目前用于与仿真进行接口的最新文档。
- 四足机器人环境受到 [Google 的 motion-imitation 仓库](https://github.com/google-research/motion_imitation) 的启发，基于[这篇论文](https://xbpeng.github.io/projects/Robotic_Imitation/2020_Robotic_Imitation.pdf)。
- 强化学习算法来源于 [stable-baselines3](https://github.com/DLR-RM/stable-baselines3)。更多示例可参考 [ray[rllib]](https://github.com/ray-project/ray) 和 [spinningup](https://github.com/openai/spinningup)。

## 概念资源
CPG 和 RL 框架基于以下论文：
- G. Bellegarda 和 A. Ijspeert，“CPG-RL：学习用于四足机器人运动的中央模式发生器”，发表于 IEEE Robotics and Automation Letters, 2022，doi: 10.1109/LRA.2022.3218167。[IEEE](https://ieeexplore.ieee.org/abstract/document/9932888)，[arxiv](https://arxiv.org/abs/2211.00458)
- G. Bellegarda, Y. Chen, Z. Liu 和 Q. Nguyen，“通过深度强化学习实现四足机器人高速稳健的奔跑”，发表于 2022 IEEE/RSJ International Conference on Intelligent Robots and Systems, 2022。[arxiv](https://arxiv.org/abs/2103.06484)

## 提示
- 如果仿真速度非常慢，删除对 time.sleep() 的调用并在 [quadruped_gym_env.py](./env/quadruped_gym_env.py) 中禁用相机重置。
- 可以在 [quadruped_gym_env.py](./env/quadruped_gym_env.py) 的 `_render_step_helper()` 函数中修改相机视角以跟随机器人。
