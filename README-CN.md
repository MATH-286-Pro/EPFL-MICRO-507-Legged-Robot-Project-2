# 中文说明文件


## 1.环境配置
- 1.1 添加虚拟环境 + 激活虚拟环境
    ```  
    virtualenv micro-507 --python=python3
    cd micro-507
    source Scripts/activate
    cd ..
    ```
- 1.2 安装依赖
    ```
    pip install -r requirements.txt
    ```
- 1.3 可以使用如下命令退出虚拟环境   
  ```
  deactivate
  ```


## 2.文件说明
- **涉及知识点**
  - CPG 中央模式发生器
  - Deep RL 深度强化学习


## 3.任务说明
- **run_cpg.py & hopf_network.py**
  - `hopf_network.py` 提供了一个 CPG 类的框架用于实现各种步态
  - 你需要将些映射到 `run_cpg.py` 的关节指令中
  - 然后在 quadruped_gym_env.py 类的实例中执行
- **run_sb3.py & load_sb3.py**
  - 这两个文件提供了基于 stable-baselines3 的强化学习算法训练接口
  - 你需要仔细阅读文档以了解不同算法和训练超参数
  - 当然，你也可以使用其他强化学习库


## 4.测试日志