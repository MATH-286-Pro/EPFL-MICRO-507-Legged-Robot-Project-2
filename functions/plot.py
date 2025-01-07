import matplotlib.pyplot as plt
import numpy as np

def plot_cpg_states(t, cpg_states):
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    for i in range(4):  # For each of the four legs
        ax1 = axes[i]       # Current subplot
        ax2 = ax1.twinx()   # Create a twin axis on the right

        # Left y-axis: r and r_dot
        ax1.plot(t, cpg_states[:, i, 0], label=f'Leg {i+1} r', color='blue', linestyle='solid')      # r
        ax1.plot(t, cpg_states[:, i, 2], label=f'Leg {i+1} r_dot', color='blue', linestyle='dashed')  # r_dot
        ax1.set_ylabel('r / r_dot', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Right y-axis: theta and theta_dot
        ax2.plot(t, cpg_states[:, i, 1], label=f'Leg {i+1} theta', color='red', linestyle='solid')     # theta
        ax2.plot(t, cpg_states[:, i, 3], label=f'Leg {i+1} theta_dot', color='red', linestyle='dashed')  # theta_dot
        ax2.set_ylabel('theta / theta_dot', color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        # Legend
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')

        # Add title
        ax1.set_title(f'Leg {i+1} Foot Position Over Time')

        # Add grid
        ax1.grid(True)

    # Add overall x-axis label
    fig.text(0.5, 0.04, 'Time (s)', ha='center', fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])  # Automatically adjust subplot layout
    plt.show()
    pass


def plot_foot_comparison(t, foot_positions_real, foot_positions_des):
    fig, ax = plt.subplots(figsize=(10, 6))

    leg_i = 0

    # Plot comparison in the X direction
    ax.plot(t, foot_positions_real[:, leg_i, 0], label='Real X', color='blue', linestyle='solid')
    ax.plot(t, foot_positions_des[:, leg_i, 0], label='Desired X', color='blue', linestyle='dashed')

    # Plot comparison in the Y direction
    ax.plot(t, foot_positions_real[:, leg_i, 1], label='Real Y', color='green', linestyle='solid')
    ax.plot(t, foot_positions_des[:, leg_i, 1], label='Desired Y', color='green', linestyle='dashed')

    # Plot comparison in the Z direction
    ax.plot(t, foot_positions_real[:, leg_i, 2], label='Real Z', color='red', linestyle='solid')
    ax.plot(t, foot_positions_des[:, leg_i, 2], label='Desired Z', color='red', linestyle='dashed')

    # Add legend
    ax.legend()

    # Add axis labels and title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Foot Position (m)')
    ax.set_title('Foot Positions Real vs Desired')

    # Add grid
    ax.grid(True)

    # Display the plot
    plt.tight_layout()
    plt.show()

    pass


def plot_real_vs_desired(t, real_data, desired_data, labels, y_label, title, colors=None, figsize=(10, 6)):
    """
    Plot comparison between real values and desired values.

    Parameters:
    - t: Time series (array-like)
    - real_data: Actual values matrix (shape: [N, D], N is the number of time steps, D is the dimension)
    - desired_data: Desired values matrix (shape: [N, D], N is the number of time steps, D is the dimension)
    - labels: Labels for each dimension (list of str)
    - y_label: Y-axis label (str)
    - title: Chart title (str)
    - colors: Colors for each dimension (list of str), defaults to None to use default colors
    - figsize: Figure size (tuple)
    """
    fig, ax = plt.subplots(figsize=figsize)

    leg_i = 0

    # If colors are not specified, use default colors
    if colors is None:
        colors = ['blue', 'green', 'red', 'orange', 'purple', 'cyan']

    for i in range(3):
        color = colors[i % len(colors)]  # Cycle through colors
        ax.plot(t, real_data[:, leg_i, i], label=f'Real {labels[i]}', color=color, linestyle='solid')
        ax.plot(t, desired_data[:, leg_i, i], label=f'Desired {labels[i]}', color=color, linestyle='dashed')

    # Add legend
    ax.legend()

    # Add axis labels and title
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # Add grid
    ax.grid(True)

    # Display the plot
    plt.tight_layout()
    plt.show()


def plot_base_velocity(t, base_velocities, figsize=(10, 6), window_size=1000):
    """
    Plot base velocities, including original and smoothed velocities.

    Parameters:
    - t: Time series (array-like)
    - base_velocities: Base velocities matrix (shape: [N, 2])
    - window_size: Smoothing window size (int), default is 1000
    """
    # Compute speed magnitude
    speed = np.sqrt(base_velocities[:, 0]**2 + base_velocities[:, 1]**2)

    # Moving average smoothing
    smoothed_speed = np.convolve(speed, np.ones(window_size)/window_size, mode='same')

    # Plot
    fig = plt.figure(figsize=figsize)
    plt.plot(t, speed, label='Original Base Speed', color='blue', alpha=0.6)  # Original speed
    plt.plot(t, smoothed_speed, label='Smoothed Base Speed', color='red', linestyle='dashed')  # Smoothed speed
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Base Speed (m/s)')
    plt.title('Base Speed Over Time (Original vs Smoothed)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_RollPitch(t, base_RollPitchYaw, figsize=(10, 6)):
    fig, ax = plt.subplots(nrows=2, figsize=figsize)

    # Plot Roll on the first subplot
    ax[0].plot(t, base_RollPitchYaw[:, 0], label='Roll', color='blue', linestyle='solid')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Angle (rad)')
    ax[0].set_title('Roll')
    ax[0].grid(True)
    ax[0].legend()

    # Plot Pitch on the second subplot
    ax[1].plot(t, base_RollPitchYaw[:, 1], label='Pitch', color='green', linestyle='solid')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Angle (rad)')
    ax[1].set_title('Pitch')
    ax[1].grid(True)
    ax[1].legend()

    # Adjust layout
    plt.tight_layout()
    plt.show()


## Add testing
def plot_data(t, data_list, base_positions, base_velocities, base_RollPitchYaw, energy, figsize=(10, 8)):
    rad2deg = 180 / np.pi

    # Determine the number of subplots and figure layout
    n_plots = len(data_list)
    fig, ax = plt.subplots(nrows=n_plots, figsize=figsize)

    # Ensure ax is a list (even if there's only one subplot)
    if n_plots == 1:
        ax = [ax]

    # Iterate over the input data_list and plot the corresponding data
    for i, data in enumerate(data_list):
        if data == 'energy':

            window_size = 20
            smoothed_energy = np.convolve(energy, np.ones(window_size)/window_size, mode='same')

            ax[i].plot(t, energy, label='Energy', color='blue', alpha=0.6, linestyle='solid')
            ax[i].plot(t, smoothed_energy, label='Smoothed Energy', color='purple', linestyle='dashed')
            ax[i].set_xlabel('Time (s)')
            ax[i].set_ylabel('Energy (J/timestep)')
            ax[i].set_title('Power Consumption Over Time')
            ax[i].grid(True)
            ax[i].legend()

        elif data == 'roll':
            ax[i].plot(t, base_RollPitchYaw[:, 0]*rad2deg, label='Roll', color='blue', linestyle='solid')
            ax[i].set_xlabel('Time (s)')
            ax[i].set_ylabel('Angle (degrees)')
            ax[i].set_title('Roll Over Time')
            ax[i].grid(True)
            ax[i].legend()

        elif data == 'pitch':
            ax[i].plot(t, base_RollPitchYaw[:, 1]*rad2deg, label='Pitch', color='green', linestyle='solid')
            ax[i].set_xlabel('Time (s)')
            ax[i].set_ylabel('Angle (degrees)')
            ax[i].set_title('Pitch Over Time')
            ax[i].grid(True)
            ax[i].legend()

        elif data == 'yaw':
            ax[i].plot(t, base_RollPitchYaw[:, 2]*rad2deg, label='Yaw', color='red', linestyle='solid')
            ax[i].set_xlabel('Time (s)')
            ax[i].set_ylabel('Angle (degrees)')
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
