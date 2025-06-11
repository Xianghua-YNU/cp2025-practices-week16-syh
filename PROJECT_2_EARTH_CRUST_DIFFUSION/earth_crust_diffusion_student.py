import numpy as np
import matplotlib.pyplot as plt

def solve_earth_crust_diffusion():
    """
    实现显式差分法求解地壳热扩散问题
    
    返回:
        tuple: (depth_array, temperature_matrix)
        depth_array: 深度坐标数组 (m)
        temperature_matrix: 温度场矩阵 (°C)
    
    物理背景: 模拟地壳中温度随深度和时间的周期性变化
    数值方法: 显式差分格式
    
    实现步骤:
    1. 设置物理参数和网格参数
    2. 初始化温度场
    3. 应用边界条件
    4. 实现显式差分格式
    5. 返回计算结果
    """
    # 设置物理参数
    D = 0.1  # 热扩散率，单位 m^2/day
    A = 10.0  # 地表平均温度，单位 °C
    B = 12.0  # 地表温度振幅，单位 °C
    tau = 365.0  # 周期，单位 day
    T_bottom = 11.0  # 20米深处温度，单位 °C
    
    # 设置网格参数
    L = 20.0  # 总深度，单位 m
    N = 21  # 深度方向网格点数（匹配测试要求的21个深度点）
    dx = L / (N - 1)  # 空间步长
    
    # 时间参数
    total_days = 365  # 模拟1年
    dt = 1.0  # 时间步长为1天
    # 检查稳定性条件：dt <= dx^2 / (2*D)
    stability_limit = dx**2 / (2 * D)
    if dt > stability_limit:
        print(f"警告：时间步长太大，可能导致数值不稳定。建议使用 dt <= {stability_limit:.4f}")
    
    M = 365  # 时间步数（匹配测试要求的366个时间点，包括初始时刻）
    
    # 初始化温度场 - 确保形状正确
    T = np.zeros((N, M + 1))  # 深度在前，时间在后
    T[:, 0] = A  # 初始温度设为地表平均温度
    
    # 设置边界条件 - 确保索引正确
    for m in range(M + 1):
        t = m * dt  # 当前时间
        T[0, m] = A + B * np.sin(2 * np.pi * t / tau)  # 地表温度 (深度0)
        T[-1, m] = T_bottom  # 底部温度 (深度20m)
    
    # 显式差分格式 - 确保计算正确
    alpha = D * dt / dx**2  # 稳定性参数
    
    for m in range(M):
        for i in range(1, N - 1):
            T[i, m + 1] = T[i, m] + alpha * (T[i + 1, m] - 2 * T[i, m] + T[i - 1, m])
    
    # 计算深度数组
    depth = np.linspace(0, L, N)
    
    # 转置矩阵以匹配测试用例的预期输出格式 (21, 366)
    T = T.T
    
    # 验证结果形状
    assert T.shape == (21, 366), f"温度矩阵形状错误: {T.shape}, 应为 (21, 366)"
    
    return depth, T

if __name__ == "__main__":
    # 运行模拟
    depth, T = solve_earth_crust_diffusion()
    print(f"计算完成，温度场形状: {T.shape}")
    
    # 可视化结果
    plt.figure(figsize=(10, 6))
    
    # 选择一年中的4个时间点（代表四季）
    days_per_season = 365 // 4
    
    seasons = [0, days_per_season, 2*days_per_season, 3*days_per_season]
    season_names = ["冬季", "春季", "夏季", "秋季"]
    
    for i, day in enumerate(seasons):
        plt.plot(depth, T[day, :], label=season_names[i])
    
    plt.xlabel("深度 (m)")
    plt.ylabel("温度 (°C)")
    plt.title("地壳温度随深度的变化（一年四季）")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 分析温度振幅随深度的变化
    # 计算每个深度的温度振幅和相位
    amplitudes = np.zeros(len(depth))
    phases = np.zeros(len(depth))
    
    for i in range(len(depth)):
        # 使用FFT分析周期性
        temp_data = T[:, i]
        fft_result = np.fft.rfft(temp_data)
        # 找到主频率分量（一年周期）
        main_freq_idx = 1  # 假设主频率是第一个非零频率
        amplitudes[i] = 2 * np.abs(fft_result[main_freq_idx]) / len(temp_data)
        phases[i] = np.angle(fft_result[main_freq_idx])
    
    # 绘制振幅随深度的变化
    plt.figure(figsize=(10, 6))
    plt.plot(depth, amplitudes)
    plt.xlabel("深度 (m)")
    plt.ylabel("温度振幅 (°C)")
    plt.title("温度振幅随深度的衰减")
    plt.grid(True)
    plt.show()
    
    # 绘制相位随深度的变化
    plt.figure(figsize=(10, 6))
    plt.plot(depth, phases)
    plt.xlabel("深度 (m)")
    plt.ylabel("相位 (rad)")
    plt.title("温度相位随深度的变化")
    plt.grid(True)
    plt.show()
