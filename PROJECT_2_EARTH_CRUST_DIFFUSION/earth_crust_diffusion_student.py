import numpy as np
import matplotlib.pyplot as plt

# 物理常数
D = 0.1  # 热扩散率 (m^2/day)
A = 10.0  # 年平均地表温度 (°C)
B = 12.0  # 地表温度振幅 (°C)
TAU = 365.0  # 年周期 (days)
T_BOTTOM = 11.0  # 20米深处温度 (°C)
T_INITIAL = 10.0  # 初始温度 (°C)
DEPTH_MAX = 20.0  # 最大深度 (m)

def solve_earth_crust_diffusion(h=1.0, a=1.0, M=21, N=366, years=1):
    """
    求解地壳热扩散方程 (显式差分格式)
    
    参数:
        h (float): 空间步长 (m)
        a (float): 时间步长比例因子
        M (int): 深度方向网格点数
        N (int): 时间步数
        years (int): 总模拟年数
    
    返回:
        tuple: (depth_array, temperature_matrix)
            - depth_array (ndarray): 深度数组 (m)
            - temperature_matrix (ndarray): 温度矩阵 [depth, time]
    """
    # 计算时间步长和稳定性参数
    dt = a**2 / (2 * D)  # 时间步长
    r = D * dt / h**2    # 稳定性参数
    
    print(f"空间步长 h = {h:.2f} m")
    print(f"时间步长 dt = {dt:.2f} days")
    print(f"稳定性参数 r = {r:.4f} (应小于0.5以保证数值稳定性)")
    
    # 初始化温度矩阵 [depth, time]
    T = np.zeros((M, N))
    T[:, 0] = T_INITIAL  # 设置初始温度
    
    # 应用边界条件
    for j in range(1, N):
        # 地表边界条件 (深度0)
        T[0, j] = A + B * np.sin(2 * np.pi * j / TAU)
        
        # 底部边界条件 (深度20m)
        T[-1, j] = T_BOTTOM
    
    # 显式差分格式求解
    for j in range(0, N-1):
        # 对内部点应用差分格式
        for i in range(1, M-1):
            T[i, j+1] = T[i, j] + r * (T[i+1, j] - 2*T[i, j] + T[i-1, j])
    
    # 创建深度数组
    depth = np.linspace(0, DEPTH_MAX, M)
    
    return depth, T

def plot_seasonal_profiles(depth, temperature, seasons=None):
    """
    绘制季节性温度轮廓
    
    参数:
        depth (ndarray): 深度数组
        temperature (ndarray): 温度矩阵
        seasons (list): 季节时间点 (days)，默认为四季
    """
    if seasons is None:
        seasons = [0, 90, 180, 270]  # 春夏秋冬四个季节
    
    season_names = ["冬季", "春季", "夏季", "秋季"]
    
    plt.figure(figsize=(10, 6))
    
    # 绘制各季节的温度轮廓
    for i, day in enumerate(seasons):
        plt.plot(depth, temperature[:, day], 
                label=f'{season_names[i]} (第{day}天)', linewidth=2)
    
    plt.xlabel('深度 (m)')
    plt.ylabel('温度 (°C)')
    plt.title('地壳温度随深度的变化（四季）')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    return plt

def analyze_amplitude_phase(depth, temperature):
    """
    分析温度振幅和相位随深度的变化
    
    参数:
        depth (ndarray): 深度数组
        temperature (ndarray): 温度矩阵
    
    返回:
        tuple: (amplitudes, phases)
            - amplitudes (ndarray): 各深度的温度振幅
            - phases (ndarray): 各深度的温度相位
    """
    # 计算每个深度的温度振幅和相位
    amplitudes = np.zeros(len(depth))
    phases = np.zeros(len(depth))
    
    for i in range(len(depth)):
        # 使用FFT分析周期性
        temp_data = temperature[i, :]
        fft_result = np.fft.rfft(temp_data)
        
        # 找到主频率分量（一年周期）
        main_freq_idx = 1  # 第一个非零频率对应一年周期
        amplitudes[i] = 2 * np.abs(fft_result[main_freq_idx]) / len(temp_data)
        phases[i] = np.angle(fft_result[main_freq_idx])
    
    return amplitudes, phases

def plot_amplitude_phase(depth, amplitudes, phases):
    """
    绘制温度振幅和相位随深度的变化
    
    参数:
        depth (ndarray): 深度数组
        amplitudes (ndarray): 各深度的温度振幅
        phases (ndarray): 各深度的温度相位
    """
    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 绘制振幅随深度的变化
    ax1.plot(depth, amplitudes)
    ax1.set_xlabel('深度 (m)')
    ax1.set_ylabel('温度振幅 (°C)')
    ax1.set_title('温度振幅随深度的衰减')
    ax1.grid(True)
    
    # 绘制相位随深度的变化
    ax2.plot(depth, phases)
    ax2.set_xlabel('深度 (m)')
    ax2.set_ylabel('相位 (rad)')
    ax2.set_title('温度相位随深度的变化')
    ax2.grid(True)
    
    plt.tight_layout()
    return plt

if __name__ == "__main__":
    # 运行模拟
    depth, T = solve_earth_crust_diffusion(h=1.0, a=1.0, M=21, N=366, years=1)
    
    # 绘制季节性温度轮廓
    plt1 = plot_seasonal_profiles(depth, T)
    plt1.show()
    
    # 分析并绘制温度振幅和相位
    amplitudes, phases = analyze_amplitude_phase(depth, T)
    plt2 = plot_amplitude_phase(depth, amplitudes, phases)
    plt2.show()
    
    # 输出结果
    print(f"计算完成，温度场形状: {T.shape}")
    print(f"地表温度振幅: {amplitudes[0]:.2f} °C")
    print(f"20米深处温度振幅: {amplitudes[-1]:.2f} °C")
