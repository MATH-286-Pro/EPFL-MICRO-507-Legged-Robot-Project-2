import matplotlib.pyplot as plt
import numpy as np

def plot_cpg_states(t, cpg_states):

  fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

  for i in range(4):     # 针对四条腿
      ax1 = axes[i]      # 当前的子图
      ax2 = ax1.twinx()  # 创建右轴
      
      # 左侧坐标轴: r 和 r_dot
      ax1.plot(t, cpg_states[:, i, 0], label=f'Leg {i+1} r', color='blue', linestyle='solid')      # r
      ax1.plot(t, cpg_states[:, i, 2], label=f'Leg {i+1} r_dot', color='blue', linestyle='dashed')  # r_dot
      ax1.set_ylabel('r / r_dot', color='blue')
      ax1.tick_params(axis='y', labelcolor='blue')
      
      # 右侧坐标轴: theta 和 theta_dot
      ax2.plot(t, cpg_states[:, i, 1], label=f'Leg {i+1} theta', color='red', linestyle='solid')     # theta
      ax2.plot(t, cpg_states[:, i, 3], label=f'Leg {i+1} theta_dot', color='red', linestyle='dashed')  # theta_dot
      ax2.set_ylabel('theta / theta_dot', color='red')
      ax2.tick_params(axis='y', labelcolor='red')
      
      # 图例
      ax1.legend(loc='upper left')
      ax2.legend(loc='upper right')
      
      # 添加标题
      ax1.set_title(f'Leg {i+1} Foot Position Over Time')
      
      # 添加网格
      ax1.grid(True)

  # 添加整体的 x 轴标签
  fig.text(0.5, 0.04, 'Time (s)', ha='center', fontsize=12)
  fig.tight_layout(rect=[0, 0.05, 1, 0.95])  # 自动调整子图布局
  plt.show()
  pass





def plot_foot_comparison(t, foot_positions_real, foot_positions_des):
  fig, ax = plt.subplots(figsize=(10, 6))

  leg_i = 0

  # 绘制 X 方向的对比
  ax.plot(t, foot_positions_real[:,leg_i, 0], label='Real X', color='blue', linestyle='solid')
  ax.plot(t, foot_positions_des [:,leg_i, 0], label='Desired X', color='blue', linestyle='dashed')

  # 绘制 Y 方向的对比
  ax.plot(t, foot_positions_real[:,leg_i, 1], label='Real Y', color='green', linestyle='solid')
  ax.plot(t, foot_positions_des [:,leg_i, 1], label='Desired Y', color='green', linestyle='dashed')

  # 绘制 Z 方向的对比
  ax.plot(t, foot_positions_real[:,leg_i, 2], label='Real Z', color='red', linestyle='solid')
  ax.plot(t, foot_positions_des [:,leg_i, 2], label='Desired Z', color='red', linestyle='dashed')

  # 添加图例
  ax.legend()

  # 添加轴标签和标题
  ax.set_xlabel('Time (s)')
  ax.set_ylabel('Foot Position (m)')
  ax.set_title('Foot Positions Real vs Desired')

  # 添加网格
  ax.grid(True)

  # 显示图像
  plt.tight_layout()
  plt.show()

  pass




def plot_real_vs_desired(t, real_data, desired_data, labels, y_label, title, colors=None, figsize=(10, 6)):
    """
    绘制真实值与期望值的对比图
    
    参数:
    - t: 时间序列 (array-like)
    - real_data: 实际值矩阵 (shape: [N, D], N为时间步数，D为维度)
    - desired_data: 期望值矩阵 (shape: [N, D], N为时间步数，D为维度)
    - labels: 每个维度的标签 (list of str)
    - y_label: y轴标签 (str)
    - title: 图表标题 (str)
    - colors: 每个维度的颜色 (list of str)，默认为None，使用默认配色
    - figsize: 图表大小 (tuple)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    leg_i = 0

    # 如果未指定颜色，使用默认颜色
    if colors is None:
        colors = ['blue', 'green', 'red', 'orange', 'purple', 'cyan']
    
    for i in range(3):
        color = colors[i % len(colors)]  # 循环选择颜色
        ax.plot(t, real_data   [:,leg_i, i], label=f'Real {labels[i]}', color=color, linestyle='solid')
        ax.plot(t, desired_data[:,leg_i, i], label=f'Desired {labels[i]}', color=color, linestyle='dashed')
    
    # 添加图例
    ax.legend()
    
    # 添加轴标签和标题
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(y_label)
    ax.set_title(title)
    
    # 添加网格
    ax.grid(True)
    
    # 显示图像
    plt.tight_layout()
    plt.show()





def plot_base_velocity(t, base_velocities, figsize=(10, 6) , window_size=1000):
    """
    绘制基座速度，包括原始速度和平滑后的速度。
    
    参数:
    - t: 时间序列 (array-like)
    - base_velocities: 基座速度矩阵 (shape: [N, 2])
    - window_size: 平滑窗口大小 (int)，默认为10
    """
    # 计算速度平方和
    speed = np.sqrt(base_velocities[:, 0]**2 + base_velocities[:, 1]**2)
    
    # 移动平均平滑处理
    smoothed_speed = np.convolve(speed, np.ones(window_size)/window_size, mode='same')
    
    # 绘图
    fig = plt.figure(figsize=figsize)
    plt.plot(t, speed, label='Original Base Speed', color='blue', alpha=0.6)  # 原始速度
    plt.plot(t, smoothed_speed, label='Smoothed Base Speed', color='red', linestyle='dashed')  # 平滑后的速度
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Base Speed (m/s)')
    plt.title('Base Speed Over Time (Original vs Smoothed)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt

def plot_RollPitch(t, base_RollPitchYaw, figsize=(10, 6)):
    fig, ax = plt.subplots(nrows=2, figsize=figsize)
    
    # 绘制 Roll 在第一张图
    ax[0].plot(t, base_RollPitchYaw[:, 0], label='Roll', color='blue', linestyle='solid')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Angle (rad)')
    ax[0].set_title('Roll')
    ax[0].grid(True)
    ax[0].legend()
    
    # 绘制 Pitch 在第二张图
    ax[1].plot(t, base_RollPitchYaw[:, 1], label='Pitch', color='green', linestyle='solid')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Angle (rad)')
    ax[1].set_title('Pitch')
    ax[1].grid(True)
    ax[1].legend()
    
    # 调整布局
    plt.tight_layout()
    plt.show()



## 添加测试
def plot_data(t, data_list, base_positions, base_velocities, base_RollPitchYaw, energy, figsize=(10, 8)):
    rad2deg = 180 / np.pi

    # 创建子图的数量和图形布局
    n_plots = len(data_list)
    fig, ax = plt.subplots(nrows=n_plots, figsize=figsize)

    # 确保 ax 是列表（即使只有一个子图）
    if n_plots == 1:
        ax = [ax]

    # 遍历输入的 data_list 并绘制相应的数据
    for i, data in enumerate(data_list):
        if data == 'energy':

            window_size = 20
            smoothed_energy = np.convolve(energy, np.ones(window_size)/window_size, mode='same')

            ax[i].plot(t, energy, label='Energy', color='blue',alpha = 0.6, linestyle='solid')
            ax[i].plot(t, smoothed_energy, label='Smoothed Energy', color='purple', linestyle='dashed')
            ax[i].set_xlabel('Time (s)')
            ax[i].set_ylabel('Energy (J/timestep)')
            ax[i].set_title('Power consumtion Over Time')
            ax[i].grid(True)
            ax[i].legend()
        
        elif data == 'roll':
            ax[i].plot(t, base_RollPitchYaw[:, 0]*rad2deg, label='Roll', color='blue', linestyle='solid')
            ax[i].set_xlabel('Time (s)')
            ax[i].set_ylabel('Angle (degree)')
            ax[i].set_title('Roll Over Time')
            ax[i].grid(True)
            ax[i].legend()

        elif data == 'pitch':
            ax[i].plot(t, base_RollPitchYaw[:, 1]*rad2deg, label='Pitch', color='green', linestyle='solid')
            ax[i].set_xlabel('Time (s)')
            ax[i].set_ylabel('Angle (degree)')
            ax[i].set_title('Pitch Over Time')
            ax[i].grid(True)
            ax[i].legend()

        elif data == 'yaw':
            ax[i].plot(t, base_RollPitchYaw[:, 2]*rad2deg, label='Yaw', color='red', linestyle='solid')
            ax[i].set_xlabel('Time (s)')
            ax[i].set_ylabel('Angle (degree)')
            ax[i].set_title('Yaw Over Time')
            ax[i].grid(True)
            ax[i].legend()

        elif data == 'x_velocity':
            ax[i].plot(t, base_velocities[:, 0], label='X Velocity', color='purple', linestyle='solid')
            ax[i].set_xlabel('Time (s)')
            ax[i].set_ylabel('Velocity (m/s)')
            ax[i].set_title('X Velocity Over Time')
            ax[i].grid(True)
            ax[i].legend()

    plt.tight_layout()
    plt.show()


