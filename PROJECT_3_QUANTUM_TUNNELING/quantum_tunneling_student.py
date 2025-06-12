import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import linalg

class QuantumTunnelingSolver:
    def __init__(self, Nx=220, Nt=300, x0=40, k0=0.5, d=10, barrier_width=3, barrier_height=1.0):
        """
        初始化量子隧穿求解器
        
        参数:
            Nx (int): 空间网格点数
            Nt (int): 时间步数
            x0 (float): 波包初始位置
            k0 (float): 波包初始动量
            d (float): 波包宽度参数
            barrier_width (float): 势垒宽度
            barrier_height (float): 势垒高度
        """
        # 空间和时间参数
        self.Nx = Nx
        self.Nt = Nt
        self.x = np.arange(self.Nx)  # 简化的空间网格
        
        # 波包参数
        self.x0 = x0
        self.k0 = k0
        self.d = d
        
        # 势垒参数
        self.barrier_width = barrier_width
        self.barrier_height = barrier_height
        
        # 初始化波函数和势函数
        self.V = self.initialize_potential()
        
        # 波函数历史记录
        self.B = np.zeros((self.Nx, self.Nt), dtype=complex)  # 波函数历史
        self.C = np.zeros((self.Nx, self.Nt), dtype=complex)  # 辅助计算
        self.B[:, 0] = self.initialize_wavefunction()
        
        # 系数矩阵
        self.A = self.build_coefficient_matrix()
        
        # 概率守恒验证
        self.total_prob = np.zeros(self.Nt)
        self.total_prob[0] = np.sum(np.abs(self.B[:, 0])**2)
    
    def initialize_wavefunction(self):
        """初始化高斯波包"""
        # 使用与答案代码一致的波包定义
        return np.exp(1j * self.k0 * self.x) * np.exp(-(self.x - self.x0)**2 * np.log10(2) / self.d**2)
    
    def initialize_potential(self):
        """初始化势函数"""
        V = np.zeros(self.Nx)
        # 在中心位置设置势垒
        V[self.Nx//2:self.Nx//2+self.barrier_width] = self.barrier_height
        return V
    
    def build_coefficient_matrix(self):
        """构建Crank-Nicolson格式的三对角系数矩阵"""
        # 使用与答案代码一致的系数矩阵构建方法
        A = np.diag(-2 + 2j - self.V) + np.diag(np.ones(self.Nx-1), 1) + np.diag(np.ones(self.Nx-1), -1)
        return A
    
    def solve_schrodinger(self):
        """求解薛定谔方程，计算波函数随时间的演化"""
        # 时间步进
        for t in range(self.Nt-1):
            # 使用与答案代码一致的求解方法
            self.C[:, t+1] = 4j * linalg.solve(self.A, self.B[:, t])
            self.B[:, t+1] = self.C[:, t+1] - self.B[:, t]
            
            # 验证概率守恒
            self.total_prob[t+1] = np.sum(np.abs(self.B[:, t+1])**2)
        
        return self.x, self.V, self.B, self.C
    
    def calculate_transmission_reflection(self):
        """计算透射系数和反射系数"""
        barrier_position = self.Nx // 2
        
        # 计算透射区域概率
        transmitted_prob = np.sum(np.abs(self.B[barrier_position+self.barrier_width:, -1])**2)
        
        # 计算反射区域概率
        reflected_prob = np.sum(np.abs(self.B[:barrier_position, -1])**2)
        
        # 计算总概率
        total_prob = np.sum(np.abs(self.B[:, -1])**2)
        
        # 计算透射系数和反射系数
        transmission = transmitted_prob / total_prob
        reflection = reflected_prob / total_prob
        
        return transmission, reflection
    
    def verify_probability_conservation(self):
        """验证概率守恒"""
        return self.total_prob
    
    def plot_potential(self):
        """绘制势函数分布图"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.x, self.V, 'k-', linewidth=2)
        plt.xlabel('位置 x')
        plt.ylabel('势能 V(x)')
        plt.title('势能函数分布')
        plt.grid(True)
        plt.show()
    
    def plot_evolution(self, time_indices=None):
        """绘制波函数概率密度随时间的演化"""
        if time_indices is None:
            time_indices = [0, self.Nt//4, self.Nt//2, 3*self.Nt//4, self.Nt-1]
        
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()
        
        # 添加总体标题
        fig.suptitle(f'量子隧穿演化 - 势垒宽度: {self.barrier_width}, 势垒高度: {self.barrier_height}', 
                     fontsize=14, fontweight='bold')
        
        for i, t_idx in enumerate(time_indices):
            if i < len(axes):
                ax = axes[i]
                
                # 绘制概率密度
                prob_density = np.abs(self.B[:, t_idx])**2
                ax.plot(self.x, prob_density, 'b-', linewidth=2, 
                       label=f'|ψ|² at t={t_idx}')
                
                # 绘制势函数
                ax.plot(self.x, self.V, 'k-', linewidth=2, 
                       label=f'势垒 (宽度={self.barrier_width}, 高度={self.barrier_height})')
                
                ax.set_xlabel('位置')
                ax.set_ylabel('概率密度')
                ax.set_title(f'时间步: {t_idx}')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # 移除未使用的子图
        for i in range(len(time_indices), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.show()
    
    def animate_probability_density(self, interval=20, save=False, filename='quantum_tunneling.gif'):
        """创建波函数概率密度演化的动画"""
        fig = plt.figure(figsize=(10, 6))
        plt.axis([0, self.Nx, 0, np.max(self.V)*1.1])
        
        # 添加标题
        plt.title(f'量子隧穿动画 - 势垒宽度: {self.barrier_width}, 势垒高度: {self.barrier_height}', 
                 fontsize=12, fontweight='bold')
        plt.xlabel('位置')
        plt.ylabel('概率密度 / 势能')
        
        # 初始化曲线
        wave_line, = plt.plot([], [], 'r', lw=2, label='|ψ|²')
        barrier_line, = plt.plot(self.x, self.V, 'k', lw=2, 
                                label=f'势垒 (宽度={self.barrier_width}, 高度={self.barrier_height})')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        def update(frame):
            wave_line.set_data(self.x, np.abs(self.B[:, frame])**2)
            return wave_line, barrier_line
        
        # 创建动画
        anim = FuncAnimation(fig, update, frames=self.Nt, interval=interval)
        
        if save:
            anim.save(filename, writer='pillow', fps=30)
        
        plt.show()
        return anim
    
    def demonstrate(self):
        """演示量子隧穿现象"""
        print("量子隧穿模拟")
        print("=" * 40)
        
        # 求解方程
        print("求解薛定谔方程...")
        self.solve_schrodinger()
        T, R = self.calculate_transmission_reflection()
        
        print(f"\n势垒宽度:{self.barrier_width}, 势垒高度:{self.barrier_height} 结果")
        print(f"透射系数: {T:.4f}")
        print(f"反射系数: {R:.4f}")
        print(f"总和 (T + R): {T + R:.4f}")
        
        # 绘制演化
        print("\n绘制波函数演化...")
        self.plot_evolution()
        
        # 验证概率守恒
        total_prob = self.verify_probability_conservation()
        print(f"\n概率守恒验证:")
        print(f"初始概率: {total_prob[0]:.6f}")
        print(f"最终概率: {total_prob[-1]:.6f}")
        print(f"相对变化: {abs(total_prob[-1] - total_prob[0])/total_prob[0]*100:.4f}%")
        
        # 创建动画
        print("\n创建动画...")
        anim = self.animate_probability_density()
        plt.show()
        
        return anim


if __name__ == "__main__":
    # 创建求解器实例
    solver = QuantumTunnelingSolver()
    
    # 运行演示
    animation = solver.demonstrate()
