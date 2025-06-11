#!/usr/bin/env python3
"""
学生模板：热传导方程数值解法比较
文件：heat_equation_methods_student.py
重要：函数名称必须与参考答案一致！
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace
from scipy.integrate import solve_ivp
import scipy.linalg
import time

class HeatEquationSolver:
    """
    热传导方程求解器，实现四种不同的数值方法。
    
    求解一维热传导方程：du/dt = alpha * d²u/dx²
    边界条件：u(0,t) = 0, u(L,t) = 0
    初始条件：u(x,0) = phi(x)
    """
    
    def __init__(self, L=20.0, alpha=10.0, nx=21, T_final=25.0):
        """
        初始化热传导方程求解器。
        
        参数:
            L (float): 空间域长度 [0, L]
            alpha (float): 热扩散系数
            nx (int): 空间网格点数
            T_final (float): 最终模拟时间
        """
        self.L = L
        self.alpha = alpha
        self.nx = nx
        self.T_final = T_final
        
        # 空间网格
        self.x = np.linspace(0, L, nx)
        self.dx = L / (nx - 1)
        
        # 初始化解数组
        self.u_initial = self._set_initial_condition()
        
    def _set_initial_condition(self):
        """
        设置初始条件：u(x,0) = 1 当 10 <= x <= 11，否则为 0。
        
        返回:
            np.ndarray: 初始温度分布
        """
        # 创建零数组
        u = np.zeros(self.nx)
        
        # 设置初始条件（10 <= x <= 11 区域为1）
        mask = (self.x >= 10) & (self.x <= 11)
        u[mask] = 1.0
        
        # 应用边界条件
        u[0] = 0
        u[-1] = 0
        
        return u
    
    def solve_explicit(self, dt=0.01, plot_times=None):
        """
        使用显式有限差分法（FTCS）求解。
        
        参数:
            dt (float): 时间步长
            plot_times (list): 绘图时间点
            
        返回:
            dict: 包含时间点和温度数组的解数据
            
        物理背景: 显式差分法直接从当前时刻计算下一时刻的解
        数值方法: 使用scipy.ndimage.laplace计算空间二阶导数
        稳定性条件: r = alpha * dt / dx² <= 0.5
        
        实现步骤:
        1. 检查稳定性条件
        2. 初始化解数组和时间
        3. 时间步进循环
        4. 使用laplace算子计算空间导数
        5. 更新解并应用边界条件
        6. 存储指定时间点的解
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        # 计算稳定性参数 r = alpha * dt / dx²
        r = self.alpha * dt / (self.dx ** 2)
        
        # 检查稳定性条件 r <= 0.5
        if r > 0.5:
            print(f"警告: 显式方法可能不稳定，r = {r:.4f} > 0.5")
        
        # 初始化解数组和时间变量
        u = self.u_initial.copy()
        t = 0.0
        
        # 创建结果存储字典
        results = {'times': [], 'solutions': []}
        
        # 存储初始条件
        results['times'].append(t)
        results['solutions'].append(u.copy())
        
        # 时间步进循环
        start_time = time.time()
        while t < self.T_final:
            # 确保最后一步不超过T_final
            dt_effective = min(dt, self.T_final - t)
            r_effective = self.alpha * dt_effective / (self.dx ** 2)
            
            # 使用 laplace(u) 计算空间二阶导数
            lap = laplace(u, mode='constant', cval=0) / (self.dx ** 2)
            
            # 更新解：u += r * laplace(u)
            u += r_effective * lap
            
            # 应用边界条件
            u[0] = 0
            u[-1] = 0
            
            # 更新时间
            t += dt_effective
            
            # 在指定时间点存储解
            for time_point in plot_times:
                if abs(t - time_point) < 1e-10 or (t > time_point and abs(t - dt_effective - time_point) < 1e-10):
                    results['times'].append(t)
                    results['solutions'].append(u.copy())
                    break
        
        # 记录计算时间
        results['computation_time'] = time.time() - start_time
        results['stability_parameter'] = r
        results['method'] = 'explicit'
        
        return results
    
    def solve_implicit(self, dt=0.1, plot_times=None):
        """
        使用隐式有限差分法（BTCS）求解。
        
        参数:
            dt (float): 时间步长
            plot_times (list): 绘图时间点
            
        返回:
            dict: 包含时间点和温度数组的解数据
            
        物理背景: 隐式差分法在下一时刻求解线性方程组
        数值方法: 构建三对角矩阵系统并求解
        优势: 无条件稳定，可以使用较大时间步长
        
        实现步骤:
        1. 计算扩散数 r
        2. 构建三对角系数矩阵
        3. 时间步进循环
        4. 构建右端项
        5. 求解线性系统
        6. 更新解并应用边界条件
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        # 计算扩散数 r
        r = self.alpha * dt / (self.dx ** 2)
        
        # 构建三对角矩阵（内部节点）
        n_internal = self.nx - 2
        diagonals = np.zeros((3, n_internal))
        diagonals[0, 1:] = -r  # 上对角线
        diagonals[1, :] = 1 + 2 * r  # 主对角线
        diagonals[2, :-1] = -r  # 下对角线
        
        # 初始化解数组和结果存储
        u = self.u_initial.copy()
        t = 0.0
        results = {'times': [t], 'solutions': [u.copy()]}
        
        # 时间步进循环
        start_time = time.time()
        while t < self.T_final:
            # 确保最后一步不超过T_final
            dt_effective = min(dt, self.T_final - t)
            r_effective = self.alpha * dt_effective / (self.dx ** 2)
            
            # 更新三对角矩阵
            diagonals[1, :] = 1 + 2 * r_effective
            
            # 构建右端项（内部节点）
            rhs = u[1:-1].copy()
            
            # 使用 scipy.linalg.solve_banded 求解
            u_new_internal = scipy.linalg.solve_banded((1, 1), diagonals, rhs)
            
            # 更新解并应用边界条件
            u[1:-1] = u_new_internal
            u[0] = 0
            u[-1] = 0
            
            # 更新时间
            t += dt_effective
            
            # 在指定时间点存储解
            for time_point in plot_times:
                if abs(t - time_point) < 1e-10 or (t > time_point and abs(t - dt_effective - time_point) < 1e-10):
                    results['times'].append(t)
                    results['solutions'].append(u.copy())
                    break
        
        # 记录计算时间
        results['computation_time'] = time.time() - start_time
        results['stability_parameter'] = r
        results['method'] = 'implicit'
        
        return results
    
    def solve_crank_nicolson(self, dt=0.5, plot_times=None):
        """
        使用Crank-Nicolson方法求解。
        
        参数:
            dt (float): 时间步长
            plot_times (list): 绘图时间点
            
        返回:
            dict: 包含时间点和温度数组的解数据
            
        物理背景: Crank-Nicolson方法结合显式和隐式格式
        数值方法: 时间上二阶精度，无条件稳定
        优势: 高精度且稳定性好
        
        实现步骤:
        1. 计算扩散数 r
        2. 构建左端矩阵 A
        3. 时间步进循环
        4. 构建右端向量
        5. 求解线性系统 A * u^{n+1} = rhs
        6. 更新解并应用边界条件
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        # 计算扩散数 r
        r = self.alpha * dt / (self.dx ** 2)
        
        # 构建左端矩阵 A（内部节点）
        n_internal = self.nx - 2
        diagonals_A = np.zeros((3, n_internal))
        diagonals_A[0, 1:] = -r/2  # 上对角线
        diagonals_A[1, :] = 1 + r  # 主对角线
        diagonals_A[2, :-1] = -r/2  # 下对角线
        
        # 构建右端矩阵 B 的系数
        diagonals_B = np.zeros((3, n_internal))
        diagonals_B[0, 1:] = r/2  # 上对角线
        diagonals_B[1, :] = 1 - r  # 主对角线
        diagonals_B[2, :-1] = r/2  # 下对角线
        
        # 初始化解数组和结果存储
        u = self.u_initial.copy()
        t = 0.0
        results = {'times': [t], 'solutions': [u.copy()]}
        
        # 时间步进循环
        start_time = time.time()
        while t < self.T_final:
            # 确保最后一步不超过T_final
            dt_effective = min(dt, self.T_final - t)
            r_effective = self.alpha * dt_effective / (self.dx ** 2)
            
            # 更新矩阵系数
            diagonals_A[1, :] = 1 + r_effective
            diagonals_B[1, :] = 1 - r_effective
            
            # 构建右端向量
            rhs = diagonals_B[1, :] * u[1:-1]
            rhs[:-1] += diagonals_B[0, 1:] * u[2:-1]
            rhs[1:] += diagonals_B[2, :-1] * u[:-2]
            
            # 求解线性系统
            u_new_internal = scipy.linalg.solve_banded((1, 1), diagonals_A, rhs)
            
            # 更新解并应用边界条件
            u[1:-1] = u_new_internal
            u[0] = 0
            u[-1] = 0
            
            # 更新时间
            t += dt_effective
            
            # 在指定时间点存储解
            for time_point in plot_times:
                if abs(t - time_point) < 1e-10 or (t > time_point and abs(t - dt_effective - time_point) < 1e-10):
                    results['times'].append(t)
                    results['solutions'].append(u.copy())
                    break
        
        # 记录计算时间
        results['computation_time'] = time.time() - start_time
        results['stability_parameter'] = r
        results['method'] = 'crank_nicolson'
        
        return results
    
    def _heat_equation_ode(self, t, u_internal):
        """
        用于solve_ivp方法的ODE系统。
        
        参数:
            t (float): 当前时间
            u_internal (np.ndarray): 内部节点温度
            
        返回:
            np.ndarray: 内部节点的时间导数
            
        物理背景: 将PDE转化为ODE系统
        数值方法: 使用laplace算子计算空间导数
        
        实现步骤:
        1. 重构包含边界条件的完整解
        2. 使用laplace计算二阶导数
        3. 返回内部节点的导数
        """
        # 重构完整解向量（包含边界条件）
        u_full = np.zeros(self.nx)
        u_full[1:-1] = u_internal
        
        # 使用 laplace(u_full) / dx² 计算二阶导数
        lap = laplace(u_full, mode='constant', cval=0) / (self.dx ** 2)
        
        # 返回内部节点的时间导数：alpha * d²u/dx²
        return self.alpha * lap[1:-1]
    
    def solve_with_solve_ivp(self, method='BDF', plot_times=None):
        """
        使用scipy.integrate.solve_ivp求解。
        
        参数:
            method (str): 积分方法（'RK45', 'BDF', 'Radau'等）
            plot_times (list): 绘图时间点
            
        返回:
            dict: 包含时间点和温度数组的解数据
            
        物理背景: 将PDE转化为ODE系统求解
        数值方法: 使用高精度ODE求解器
        优势: 自适应步长，高精度
        
        实现步骤:
        1. 提取内部节点初始条件
        2. 调用solve_ivp求解ODE系统
        3. 重构包含边界条件的完整解
        4. 返回结果
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        # 提取内部节点初始条件
        u_internal0 = self.u_initial[1:-1].copy()
        
        # 调用 solve_ivp 求解
        start_time = time.time()
        sol = solve_ivp(
            fun=self._heat_equation_ode,
            t_span=(0, self.T_final),
            y0=u_internal0,
            method=method,
            t_eval=plot_times,
            rtol=1e-6,
            atol=1e-6
        )
        computation_time = time.time() - start_time
        
        # 重构包含边界条件的完整解
        times = sol.t
        solutions = []
        for i in range(len(times)):
            u_full = np.zeros(self.nx)
            u_full[1:-1] = sol.y[:, i]
            solutions.append(u_full)
        
        # 返回结果字典
        results = {
            'times': times,
            'solutions': solutions,
            'computation_time': computation_time,
            'method': method,
            'solver_method': 'solve_ivp'
        }
        
        return results
    
    def compare_methods(self, dt_explicit=0.01, dt_implicit=0.1, dt_cn=0.5, 
                       ivp_method='BDF', plot_times=None):
        """
        比较所有四种数值方法。
        
        参数:
            dt_explicit (float): 显式方法时间步长
            dt_implicit (float): 隐式方法时间步长
            dt_cn (float): Crank-Nicolson方法时间步长
            ivp_method (str): solve_ivp积分方法
            plot_times (list): 比较时间点
            
        返回:
            dict: 所有方法的结果
            
        实现步骤:
        1. 调用所有四种求解方法
        2. 记录计算时间和稳定性参数
        3. 返回比较结果
        """
        if plot_times is None:
            plot_times = [0, 1, 5, 15, 25]
            
        # 打印求解信息
        print("="*50)
        print(f"热传导方程求解参数:")
        print(f"  空间域长度 L = {self.L}")
        print(f"  热扩散系数 alpha = {self.alpha}")
        print(f"  空间网格点数 nx = {self.nx}")
        print(f"  最终模拟时间 T_final = {self.T_final}")
        print(f"  时间步长: dt_explicit = {dt_explicit}, dt_implicit = {dt_implicit}, dt_cn = {dt_cn}")
        print(f"  solve_ivp方法: {ivp_method}")
        print("="*50)
        
        # 调用四种求解方法
        print("求解中...")
        
        print("  显式方法 (FTCS)...")
        explicit_results = self.solve_explicit(dt=dt_explicit, plot_times=plot_times)
        
        print("  隐式方法 (BTCS)...")
        implicit_results = self.solve_implicit(dt=dt_implicit, plot_times=plot_times)
        
        print("  Crank-Nicolson方法...")
        cn_results = self.solve_crank_nicolson(dt=dt_cn, plot_times=plot_times)
        
        print("  solve_ivp方法...")
        ivp_results = self.solve_with_solve_ivp(method=ivp_method, plot_times=plot_times)
        
        # 打印每种方法的计算时间和稳定性参数
        print("\n计算结果比较:")
        print(f"  显式方法 (FTCS): 计算时间 = {explicit_results['computation_time']:.4f}s, 稳定性参数 r = {explicit_results['stability_parameter']:.4f}")
        print(f"  隐式方法 (BTCS): 计算时间 = {implicit_results['computation_time']:.4f}s, 稳定性参数 r = {implicit_results['stability_parameter']:.4f}")
        print(f"  Crank-Nicolson方法: 计算时间 = {cn_results['computation_time']:.4f}s, 稳定性参数 r = {cn_results['stability_parameter']:.4f}")
        print(f"  solve_ivp方法 ({ivp_method}): 计算时间 = {ivp_results['computation_time']:.4f}s")
        print("="*50)
        
        # 返回所有结果的字典
        all_results = {
            'explicit': explicit_results,
            'implicit': implicit_results,
            'crank_nicolson': cn_results,
            'solve_ivp': ivp_results,
            'plot_times': plot_times,
            'x': self.x
        }
        
        return all_results
    
    def plot_comparison(self, methods_results, save_figure=False, filename='heat_equation_comparison.png'):
        """
        绘制所有方法的比较图。
        
        参数:
            methods_results (dict): compare_methods的结果
            save_figure (bool): 是否保存图像
            filename (str): 保存的文件名
            
        实现步骤:
        1. 创建2x2子图
        2. 为每种方法绘制不同时间的解
        3. 设置图例、标签和标题
        4. 可选保存图像
        """
        # 创建 2x2 子图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        # 方法列表和对应的颜色
        methods = ['explicit', 'implicit', 'crank_nicolson', 'solve_ivp']
        method_names = ['显式方法 (FTCS)', '隐式方法 (BTCS)', 'Crank-Nicolson方法', 'solve_ivp方法']
        colors = ['red', 'blue', 'green', 'purple']
        
        # 为每种方法绘制解曲线
        for i, method in enumerate(methods):
            ax = axes[i]
            results = methods_results[method]
            
            for j, t in enumerate(results['times']):
                if t in methods_results['plot_times']:
                    ax.plot(methods_results['x'], results['solutions'][j], label=f't = {t}s')
            
            # 设置标题、标签、图例
            ax.set_title(method_names[i])
            ax.set_xlabel('位置 x')
            ax.set_ylabel('温度 T')
            ax.grid(True)
            ax.legend()
        
        # 调整布局
        plt.tight_layout()
        
        # 可选保存图像
        if save_figure:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"图像已保存为 {filename}")
        
        plt.show()
    
    def analyze_accuracy(self, methods_results, reference_method='solve_ivp'):
        """
        分析不同方法的精度。
        
        参数:
            methods_results (dict): compare_methods的结果
            reference_method (str): 参考方法
            
        返回:
            dict: 精度分析结果
            
        实现步骤:
        1. 选择参考解
        2. 计算其他方法与参考解的误差
        3. 统计最大误差和平均误差
        4. 返回分析结果
        """
        # 验证参考方法存在
        if reference_method not in methods_results:
            raise ValueError(f"参考方法 '{reference_method}' 不存在")
        
        # 获取参考解
        ref_results = methods_results[reference_method]
        ref_times = ref_results['times']
        ref_solutions = ref_results['solutions']
        
        # 方法列表
        methods = ['explicit', 'implicit', 'crank_nicolson']
        method_names = ['显式方法 (FTCS)', '隐式方法 (BTCS)', 'Crank-Nicolson方法']
        
        # 初始化误差结果
        error_results = {}
        
        # 计算各方法与参考解的误差
        for method, name in zip(methods, method_names):
            method_results = methods_results[method]
            method_times = method_results['times']
            method_solutions = method_results['solutions']
            
            # 查找时间点的对应关系
            time_indices = []
            for t in ref_times:
                # 找到最接近的时间点
                idx = np.argmin(np.abs(np.array(method_times) - t))
                time_indices.append(idx)
            
            # 计算误差
            errors = []
            for i, ref_idx in enumerate(time_indices):
                if ref_idx < len(method_solutions):
                    error = np.abs(method_solutions[ref_idx] - ref_solutions[i])
                    errors.append(error)
            
            # 统计误差指标
            if errors:
                max_errors = [np.max(err) for err in errors]
                mean_errors = [np.mean(err) for err in errors]
                
                error_results[method] = {
                    'times': ref_times,
                    'max_errors': max_errors,
                    'mean_errors': mean_errors,
                    'avg_max_error': np.mean(max_errors),
                    'avg_mean_error': np.mean(mean_errors)
                }
        
        # 打印精度分析结果
        print("\n精度分析:")
        print(f"参考方法: {reference_method}")
        print("各方法平均最大误差:")
        for method, name in zip(methods, method_names):
            if method in error_results:
                print(f"  {name}: {error_results[method]['avg_max_error']:.6e}")
        
        # 绘制误差图
        plt.figure(figsize=(12, 6))
        
        for method, name in zip(methods, method_names):
            if method in error_results:
                plt.plot(error_results[method]['times'], error_results[method]['max_errors'], 'o-', label=name)
        
        plt.xlabel('时间 (s)')
        plt.ylabel('最大误差')
        plt.title('各方法相对于参考解的最大误差')
        plt.grid(True)
        plt.legend()
        plt.yscale('log')
        plt.tight_layout()
        plt.show()
        
        return error_results


def main():
    """
    HeatEquationSolver类的演示。
    """
    # 创建求解器实例
    solver = HeatEquationSolver(L=20.0, alpha=10.0, nx=101, T_final=25.0)
    
    # 比较所有方法
    results = solver.compare_methods(
        dt_explicit=0.001,
        dt_implicit=0.1,
        dt_cn=0.5,
        ivp_method='BDF'
    )
    
    # 绘制比较图
    solver.plot_comparison(results, save_figure=True)
    
    # 分析精度
    accuracy = solver.analyze_accuracy(results)
    
    # 返回结果
    return solver, results, accuracy


if __name__ == "__main__":
    solver, results, accuracy = main()
