import time
import numpy as np
import matplotlib.pyplot as plt
from env.hopf_network import HopfNetwork
from env.quadruped_gym_env import QuadrupedGymEnv

# 强制使用 Nvidia GPU
import os
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

# 设置时间步长与初始化参数
TIME_STEP = 0.001
foot_y = 0.0838  # 这个是髋部的长度
sideSign = np.array([-1, 1, -1, 1])

# 创建环境和CPG
env = QuadrupedGymEnv(
                    on_rack=False,
                    isRLGymInterface=False,
                    time_step=TIME_STEP,
                    action_repeat=1,
                    motor_control_mode="TORQUE",
                    add_noise=False,
                    )
cpg = HopfNetwork(time_step=TIME_STEP)
cpg._set_gait("TROT")

# 初始化PID参数
kp = np.array([100, 100, 100])
kd = np.array([2, 2, 2])
kpCartesian = np.diag([500] * 3)
kdCartesian = np.diag([20] * 3)

# 仿真函数
def run_simulation(USE_JOINT_PD, USE_CARTESIAN_PD):
    TotalTime = 5
    TEST_STEPS = int(TotalTime / (TIME_STEP))
    t = np.arange(TEST_STEPS) * TIME_STEP

    # 数据记录
    foot_positions_real = np.zeros((TEST_STEPS, 4, 3))
    foot_positions_des  = np.zeros((TEST_STEPS, 4, 3))
    foot_angles_real    = np.zeros((TEST_STEPS, 4, 3))
    foot_angles_des     = np.zeros((TEST_STEPS, 4, 3))

    for j in range(TEST_STEPS):
        action = np.zeros(12)  # 初始化电机输入

        # 获取足端期望位置
        xs, zs = cpg.update()

        # 获取当前电机角度与速度
        q = env.robot.GetMotorAngles()
        dq = env.robot.GetMotorVelocities()

        for i in range(4):
            tau = np.zeros(3)
            group_index = np.arange(3 * i, 3 * i + 3)

            # 获取目标足端位置
            leg_xyz = np.array([xs[i], sideSign[i] * foot_y, zs[i]])

            # 使用逆运动学计算目标关节角度
            des_q = env.robot.ComputeInverseKinematics(legID=i, xyz_coord=leg_xyz)
            des_dq = np.zeros(3)

            # 关节PD控制
            real_q = q[group_index]
            real_dq = dq[group_index]
            if USE_JOINT_PD: tau += kp * (des_q - real_q) + kd * (des_dq - real_dq)

            # 笛卡尔PD控制
            J, real_p = env.robot.ComputeJacobianAndPosition(i)
            real_dp = J @ real_dq
            des_p = leg_xyz
            des_dp = np.zeros(3)
            if USE_CARTESIAN_PD: tau += J.T @ ((kpCartesian @ (des_p - real_p)) + (kdCartesian @ (des_dp - real_dp)))

            # 记录数据
            foot_positions_real[j, i, :] = real_p
            foot_positions_des [j, i, :] = des_p
            foot_angles_real   [j, i, :] = real_q
            foot_angles_des    [j, i, :] = des_q

            # 设置动作
            action[3 * i:3 * i + 3] = tau

        # 发送动作并执行一步仿真
        env.step(action)

    return foot_positions_real, foot_positions_des, foot_angles_real, foot_angles_des


# 绘制图形
def plot_real_vs_desired(t, real_data, desired_data, labels, y_label, title):

    colors = ['blue', 'green', 'red', 'orange', 'purple', 'cyan']

    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制 leg_i 的数据
    leg_i = 0
    for i in range(3):
        color = colors[i % len(colors)]
        ax.plot(t, real_data   [:, leg_i, i], label=f'Real {labels[0]} {leg_i+1}', color=color)
        ax.plot(t, desired_data[:, leg_i, i], label=f'Desired {labels[0]} {leg_i+1}', color=color, linestyle='--')

    ax.grid(True)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()


# 绘制 3 图形
def plot_real_vs_desired_3(t, real_data_1, desired_data_1, real_data_2, desired_data_2, real_data_3, desired_data_3, labels, y_label, type = 'Foot Positions'):
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'cyan']

    # 创建三个子图
    fig, ax = plt.subplots(nrows=3, figsize=(10,8))

    leg_i = 0

    # 绘制 Joint PD Only
    ax[0].set_title(type + ': Joint PD Only')
    for i in range(3):
        color = colors[i % len(colors)]
        ax[0].plot(t, real_data_1   [:, leg_i, i], label=f'Real {labels[0]} {i+1}', color=color)
        ax[0].plot(t, desired_data_1[:, leg_i, i], label=f'Desired {labels[0]} {i+1}', color=color, linestyle='--')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel(y_label)
    ax[0].grid(True)
    ax[0].legend()

    # 绘制 Cartesian PD Only
    ax[1].set_title(type + ': Cartesian PD Only')
    for i in range(3):
        color = colors[i % len(colors)]
        ax[1].plot(t, real_data_2   [:, leg_i, i], label=f'Real {labels[0]} {i+1}', color=color)
        ax[1].plot(t, desired_data_2[:, leg_i, i], label=f'Desired {labels[0]} {i+1}', color=color, linestyle='--')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel(y_label)
    ax[1].grid(True)
    ax[1].legend()

    # 绘制 Joint + Cartesian PD
    ax[2].set_title(type + ': Joint + Cartesian PD')
    for i in range(3):
        color = colors[i % len(colors)]
        ax[2].plot(t, real_data_3   [:, leg_i, i], label=f'Real {labels[0]} {i+1}', color=color)
        ax[2].plot(t, desired_data_3[:, leg_i, i], label=f'Desired {labels[0]} {i+1}', color=color, linestyle='--')
    ax[2].set_xlabel('Time (s)')
    ax[2].set_ylabel(y_label)
    ax[2].grid(True)
    ax[2].legend()

    # 调整布局
    plt.tight_layout()
    plt.show()

################################## 主程序 ##################################

# 运行三个不同的仿真
foot_positions_real_1, foot_positions_des_1, foot_angles_real_1, foot_angles_des_1 = run_simulation(True, False)
foot_positions_real_2, foot_positions_des_2, foot_angles_real_2, foot_angles_des_2 = run_simulation(False, True)
foot_positions_real_3, foot_positions_des_3, foot_angles_real_3, foot_angles_des_3 = run_simulation(True, True)


# # 绘制足部位置
# plot_real_vs_desired(
#     t=np.arange(len(foot_positions_real_1)) * TIME_STEP,
#     real_data=foot_positions_real_1,
#     desired_data=foot_positions_des_1,
#     labels=['X', 'Y', 'Z'],
#     y_label='Position (m)',
#     title='Foot Positions: Joint PD Only'
# )

# plot_real_vs_desired(
#     t=np.arange(len(foot_positions_real_2)) * TIME_STEP,
#     real_data=foot_positions_real_2,
#     desired_data=foot_positions_des_2,
#     labels=['X', 'Y', 'Z'],
#     y_label='Position (m)',
#     title='Foot Positions: Cartesian PD Only'
# )

# plot_real_vs_desired(
#     t=np.arange(len(foot_positions_real_3)) * TIME_STEP,
#     real_data=foot_positions_real_3,
#     desired_data=foot_positions_des_3,
#     labels=['X', 'Y', 'Z'],
#     y_label='Position (m)',
#     title='Foot Positions: Joint and Cartesian PD'
# )

# plt.show()



# 绘制所有控制策略下的足部位置
plot_real_vs_desired_3(
    t=np.arange(len(foot_positions_real_1)) * TIME_STEP,
    real_data_1=foot_positions_real_1,
    desired_data_1=foot_positions_des_1,
    real_data_2=foot_positions_real_2,
    desired_data_2=foot_positions_des_2,
    real_data_3=foot_positions_real_3,
    desired_data_3=foot_positions_des_3,
    labels=['X', 'Y', 'Z'],
    y_label='Position (m)',
)

plot_real_vs_desired_3(
    t=np.arange(len(foot_positions_real_1)) * TIME_STEP,
    real_data_1    = foot_angles_real_1,
    desired_data_1 = foot_angles_des_1,
    real_data_2    = foot_angles_real_2,
    desired_data_2 = foot_angles_des_2,
    real_data_3    = foot_angles_real_3,
    desired_data_3 = foot_angles_des_3,
    labels=['X', 'Y', 'Z'],
    y_label='Angle (rad)',
    type='Joint Angles'
)

plt.show()