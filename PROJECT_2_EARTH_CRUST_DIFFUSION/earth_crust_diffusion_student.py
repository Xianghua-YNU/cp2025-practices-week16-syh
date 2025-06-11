import numpy as np
import matplotlib.pyplot as plt

# 物理常数
D = 0.1  # 热扩散率 (m²/day)
A = 10.0  # 年平均地表温度 (°C)
B = 12.0  # 地表温度振幅 (°C)
TAU = 365.0  # 年周期 (days)
T_BOTTOM = 11.0  # 20米深处温度 (°C)
T_INITIAL = 10.0  # 初始温度 (°C)
DEPTH_MAX = 20.0  # 最大深度 (m)


def solve_earth_crust_diffusion(h=1.0, a=1.0, M=21, N=366):
    """
    求解地壳热扩散方程（显式差分格式）
    
    参数:
        h (float): 空间步长 (m)
        a (float): 时间步长比例因子
        M (int): 深度方向网格点数
        N (int): 时间步数（含初始时刻）
    
    返回:
        tuple: (depth_array, temperature_matrix)
            depth_array (ndarray): 深度数组，形状 (M,)
            temperature_matrix (ndarray): 温度矩阵，形状 (M, N)
    """
    # 计算时间步长和稳定性参数
    dt = a ** 2 / (2 * D)
    r = D * dt / (h ** 2)
    print(f"稳定性参数 r = {r:.4f}")
    
    # 初始化温度矩阵 [depth, time]
    T = np.ones((M, N)) * T_INITIAL
    T[-1, :] = T_BOTTOM  # 底部边界条件
    
    # 时间步进求解
    for j in range(1, N):
        # 地表边界条件
        T[0, j] = A + B * np.sin(2 * np.pi * j / TAU)
        
        # 显式差分格式（内部点）
        T[1:-1, j] = T[1:-1, j-1] + r * (T[2:, j-1] + T[:-2, j-1] - 2 * T[1:-1, j-1])
    
    # 生成深度数组
    depth = np.linspace(0, DEPTH_MAX, M)
    return depth, T


def plot_seasonal_profiles(depth, T, seasons=None):
    """绘制四季温度随深度变化曲线"""
    if seasons is None:
        seasons = [0, 90, 180, 270]  # 冬、春、夏、秋
    plt.figure(figsize=(10, 6))
    for i, day in enumerate(seasons):
        plt.plot(depth, T[:, day], label=f"第{day}天", linewidth=2)
    plt.xlabel("深度 (m)")
    plt.ylabel("温度 (°C)")
    plt.title("地壳温度的季节性分布")
    plt.grid(True)
    plt.legend()
    return plt


def analyze_amplitude(depth, T):
    """计算各深度温度振幅"""
    amplitudes = np.zeros(len(depth))
    for i in range(len(depth)):
        temp = T[i, :]
        fft = np.fft.rfft(temp)
        freq_idx = 1  # 年周期对应频率
        amplitudes[i] = 2 * np.abs(fft[freq_idx]) / len(temp)
    return amplitudes


if __name__ == "__main__":
    # 运行模拟（默认参数与测试匹配）
    depth, T = solve_earth_crust_diffusion(h=1.0, a=1.0, M=21, N=366)
    
    # 测试输出形状
    print(f"温度矩阵形状: {T.shape}")  # 应输出 (21, 366)
    
    # 验证边界条件
    surface_temp = T[0, :]
    bottom_temp = T[-1, :]
    print(f"地表温度范围: {surface_temp.min():.2f}~{surface_temp.max():.2f}°C")
    print(f"底部温度一致性: {np.all(bottom_temp == T_BOTTOM)}")
    
    # 绘制四季温度曲线
    plt = plot_seasonal_profiles(depth, T)
    plt.show()
    
    # 计算振幅
    amplitudes = analyze_amplitude(depth, T)
    print(f"地表温度振幅: {amplitudes[0]:.2f}°C")
    print(f"20米深度振幅: {amplitudes[-1]:.2f}°C")
