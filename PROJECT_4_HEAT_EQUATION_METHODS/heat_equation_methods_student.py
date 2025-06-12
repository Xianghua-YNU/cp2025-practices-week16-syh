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
            print(f"建议减小时间步长到 < {0.5 * self.dx**2 / self.alpha:.6f}")
        
        # 初始化解数组和时间变量
        u = self.u_initial.copy()
        t = 0.0
        
        # 创建结果存储字典
        results = {'times': [], 'solutions': [], 'method': 'Explicit FTCS'}
        
        # 存储初始条件
        if 0 in plot_times:
            results['times'].append(0.0)
            results['solutions'].append(u.copy())
        
        start_time = time.time()
        
        # 时间步进循环
        while t < self.T_final:
            # 确保最后一步不超过T_final
            dt_effective = min(dt, self.T_final - t)
            
            # 使用 laplace(u) 计算空间二阶导数
            du_dt = r * laplace(u)
            u += du_dt
            
            # 应用边界条件
            u[0] = 0
            u[-1] = 0
            
            # 更新时间
            t += dt_effective
            
            # 在指定时间点存储解
            for plot_time in plot_times:
                if abs(t - plot_time) < dt/2 and plot_time not in results['times']:
                    results['times'].append(t)
                    results['solutions'].append(u.copy())
        
        results['computation_time'] = time.time() - start_time
        results['stability_parameter'] = r
        
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
        results = {'times': [], 'solutions': [], 'method': 'Implicit BTCS'}
        
        # 存储初始条件
        if 0 in plot_times:
            results['times'].append(0.0)
            results['solutions'].append(u.copy())
        
        start_time = time.time()
        
        # 时间步进循环
        while t < self.T_final:
            # 确保最后一步不超过T_final
            dt_effective = min(dt, self.T_final - t)
            
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
            for plot_time in plot_times:
                if abs(t - plot_time) < dt/2 and plot_time not in results['times']:
                    results['times'].append(t)
                    results['solutions'].append(u.copy())
        
        results['computation_time'] = time.time() - start_time
        results['stability_parameter'] = r
        
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
        
        # 初始化解数组和结果存储
        u = self.u_initial.copy()
        t = 0.0
        results = {'times': [], 'solutions': [], 'method': 'Crank-Nicolson'}
        
        # 存储初始条件
        if 0 in plot_times:
            results['times'].append(0.0)
            results['solutions'].append(u.copy())
        
        start_time = time.time()
        
        # 时间步进循环
        while t < self.T_final:
            # 确保最后一步不超过T_final
            dt_effective = min(dt, self.T_final - t)
            
            # 构建右端向量
            u_internal = u[1:-1]
            rhs = (r/2) * u[:-2] + (1 - r) * u_internal + (r/2) * u[2:]
            
            # 求解线性系统
            u_new_internal = scipy.linalg.solve_banded((1, 1), diagonals_A, rhs)
            
            # 更新解并应用边界条件
            u[1:-1] = u_new_internal
            u[0] = 0
            u[-1] = 0
            
            # 更新时间
            t += dt_effective
            
            # 在指定时间点存储解
            for plot_time in plot_times:
                if abs(t - plot_time) < dt/2 and plot_time not in results['times']:
                    results['times'].append(t)
                    results['solutions'].append(u.copy())
        
        results['computation_time'] = time.time() - start_time
        results['stability_parameter'] = r
        
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
        u_full = np.concatenate(([0.0], u_internal, [0.0]))
        
        # 使用 laplace(u_full) / dx² 计算二阶导数
        d2u_dx2 = laplace(u_full) / (self.dx**2)
        
        # 返回内部节点的时间导数：alpha * d²u/dx²
        return self.alpha * d2u_dx2[1:-1]
    
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
        
        start_time = time.time()
        
        # 调用 solve_ivp 求解
        sol = solve_ivp(
            fun=self._heat_equation_ode,
            t_span=(0, self.T_final),
            y0=u_internal0,
            method=method,
            t_eval=plot_times,
            rtol=1e-8,
            atol=1e-10
        )
        
        computation_time = time.time() - start_time
        
        # 重构包含边界条件的完整解
        times = sol.t.tolist()
        solutions = []
        for i in range(len(times)):
            u_full = np.concatenate(([0.0], sol.y[:, i], [0.0]))
            solutions.append(u_full)
        
        # 返回结果字典
        results = {
            'times': times,
            'solutions': solutions,
            'method': f'solve_ivp ({method})',
            'computation_time': computation_time
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
            
        print("求解热传导方程使用四种不同方法...")
        print(f"计算域: [0, {self.L}], 网格点数: {self.nx}, 最终时间: {self.T_final}")
        print(f"热扩散系数: {self.alpha}")
        print("-" * 60)
        
        # 求解所有方法
        methods_results = {}
        
        # 显式方法
        print("1. 显式有限差分法 (FTCS)...")
        methods_results['explicit'] = self.solve_explicit(dt_explicit, plot_times)
        print(f"   计算时间: {methods_results['explicit']['computation_time']:.4f} s")
        print(f"   稳定性参数 r: {methods_results['explicit']['stability_parameter']:.4f}")
        
        # 隐式方法
        print("2. 隐式有限差分法 (BTCS)...")
        methods_results['implicit'] = self.solve_implicit(dt_implicit, plot_times)
        print(f"   计算时间: {methods_results['implicit']['computation_time']:.4f} s")
        print(f"   稳定性参数 r: {methods_results['implicit']['stability_parameter']:.4f}")
        
        # Crank-Nicolson方法
        print("3. Crank-Nicolson方法...")
        methods_results['crank_nicolson'] = self.solve_crank_nicolson(dt_cn, plot_times)
        print(f"   计算时间: {methods_results['crank_nicolson']['computation_time']:.4f} s")
        print(f"   稳定性参数 r: {methods_results['crank_nicolson']['stability_parameter']:.4f}")
        
        # solve_ivp方法
        print(f"4. solve_ivp方法 ({ivp_method})...")
        methods_results['solve_ivp'] = self.solve_with_solve_ivp(ivp_method, plot_times)
        print(f"   计算时间: {methods_results['solve_ivp']['computation_time']:.4f} s")
        
        print("-" * 60)
        print("所有方法求解完成!")
        
        return methods_results
    
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
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        method_names = ['explicit', 'implicit', 'crank_nicolson', 'solve_ivp']
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for idx, method_name in enumerate(method_names):
            ax = axes[idx]
            results = methods_results[method_name]
            
            # 绘制不同时间的解
            for i, (t, u) in enumerate(zip(results['times'], results['solutions'])):
                ax.plot(self.x, u, color=colors[i], label=f't = {t:.1f}', linewidth=2)
            
            ax.set_title(f"{results['method']}\n(计算时间: {results['computation_time']:.4f} s)")
            ax.set_xlabel('位置 x')
            ax.set_ylabel('温度 u(x,t)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_xlim(0, self.L)
            ax.set_ylim(-0.1, 1.1)
        
        plt.tight_layout()
        
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
        if reference_method not in methods_results:
            raise ValueError(f"参考方法 '{reference_method}' 未找到")
        
        reference = methods_results[reference_method]
        accuracy_results = {}
        
        print(f"\n精度分析 (参考: {reference['method']})")
        print("-" * 50)
        
        for method_name, results in methods_results.items():
            if method_name == reference_method:
                continue
                
            errors = []
            for i, (ref_sol, test_sol) in enumerate(zip(reference['solutions'], results['solutions'])):
                if i < len(results['solutions']):
                    error = np.linalg.norm(ref_sol - test_sol, ord=2)
                    errors.append(error)
            
            max_error = max(errors) if errors else 0
            avg_error = np.mean(errors) if errors else 0
            
            accuracy_results[method_name] = {
                'max_error': max_error,
                'avg_error': avg_error,
                'errors': errors
            }
            
            print(f"{results['method']:25} - 最大误差: {max_error:.2e}, 平均误差: {avg_error:.2e}")
        
        return accuracy_results


def main():
    """
    HeatEquationSolver类的演示。
    """
    # 创建求解器实例
    solver = HeatEquationSolver(L=20.0, alpha=10.0, nx=21, T_final=25.0)
    
    # 比较所有方法
    plot_times = [0, 1, 5, 15, 25]
    results = solver.compare_methods(
        dt_explicit=0.01,
        dt_implicit=0.1, 
        dt_cn=0.5,
        ivp_method='BDF',
        plot_times=plot_times
    )
    
    # 绘制比较图
    solver.plot_comparison(results, save_figure=True)
    
    # 分析精度
    accuracy = solver.analyze_accuracy(results, reference_method='solve_ivp')
    
    return solver, results, accuracy


if __name__ == "__main__":
    solver, results, accuracy = main()
