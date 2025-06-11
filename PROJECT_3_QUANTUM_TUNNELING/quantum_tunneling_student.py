import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import linalg

class QuantumTunnelingSolver:
    def __init__(self, x_range=(0, 220), Nx=220, Nt=300, k0=0.5, sigma=5.0, 
                 x0=40, barrier_width=3, barrier_height=1.0, barrier_position=110):
        """
        初始化量子隧穿求解器
        
        参数:
            x_range (tuple): 空间范围
            Nx (int): 空间网格点数
            Nt (int): 时间步数
            k0 (float): 波包初始动量
            sigma (float): 波包宽度
            x0 (float): 波包初始位置
            barrier_width (float): 势垒宽度
            barrier_height (float): 势垒高度
            barrier_position (float): 势垒位置
        """
        # 空间网格
        self.x_min, self.x_max = x_range
        self.Nx = Nx
        self.dx = (self.x_max - self.x_min) / (self.Nx - 1)
        self.x = np.linspace(self.x_min, self.x_max, self.Nx)
        
        # 时间参数
        self.t_max = 30.0  # 总模拟时间
        self.Nt = Nt
        self.dt = self.t_max / self.Nt
        
        # 波包参数
        self.k0 = k0
        self.sigma = sigma
        self.x0 = x0
        
        # 势垒参数
        self.barrier_width = barrier_width
        self.barrier_height = barrier_height
        self.barrier_position = barrier_position
        
        # 初始化波函数和势函数
        self.psi = self.initialize_wavefunction()
        self.V = self.initialize_potential()
        
        # 构建系数矩阵
        self.A = self.build_coefficient_matrix()
        
        # 存储波函数随时间的演化
        self.psi_history = np.zeros((self.Nx, self.Nt), dtype=complex)
        self.psi_history[:, 0] = self.psi
        
        # 波函数的实部和虚部
        self.B = np.zeros((self.Nx, self.Nt))  # 实部
        self.C = np.zeros((self.Nx, self.Nt))  # 虚部
        self.B[:, 0] = np.real(self.psi)
        self.C[:, 0] = np.imag(self.psi)
    
    def initialize_wavefunction(self):
        """初始化高斯波包"""
        # 一维高斯波包
        psi = np.exp(-(self.x - self.x0)**2 / (2 * self.sigma**2)) * \
              np.exp(1j * self.k0 * (self.x - self.x0))
        
        # 归一化
        norm = np.sqrt(np.sum(np.abs(psi)**2) * self.dx)
        psi = psi / norm
        
        return psi
    
    def initialize_potential(self):
        """初始化势函数"""
        V = np.zeros(self.Nx)
        # 矩形势垒
        barrier_start = self.barrier_position - self.barrier_width/2
        barrier_end = self.barrier_position + self.barrier_width/2
        V[(self.x >= barrier_start) & (self.x <= barrier_end)] = self.barrier_height
        return V
    
    def build_coefficient_matrix(self):
        """构建Crank-Nicolson格式的三对角系数矩阵"""
        # 三对角矩阵的对角线元素
        alpha = -2 + 2j * self.dt / (self.dx**2) - self.dt * self.V
        beta = np.ones(self.Nx-1) * (1j * self.dt / (self.dx**2))
        
        # 构建三对角矩阵
        A = np.diag(alpha) + np.diag(beta, k=1) + np.diag(beta, k=-1)
        
        # 边界条件：波函数在边界处为零
        A[0, 0] = 1
        A[0, 1] = 0
        A[-1, -1] = 1
        A[-1, -2] = 0
        
        return A
    
    def solve_schrodinger(self):
        """求解薛定谔方程，计算波函数随时间的演化"""
        # 时间步进
        for j in range(1, self.Nt):
            # 构建右侧向量
            b = 4j * self.dt / (self.dx**2) * self.psi
            
            # 边界条件
            b[0] = 0
            b[-1] = 0
            
            # 求解线性方程组得到χ
            chi = linalg.solve(self.A, b)
            
            # 更新波函数
            self.psi = chi - self.psi
            
            # 归一化
            norm = np.sqrt(np.sum(np.abs(self.psi)**2) * self.dx)
            self.psi = self.psi / norm
            
            # 存储当前时间步的波函数
            self.psi_history[:, j] = self.psi
            self.B[:, j] = np.real(self.psi)
            self.C[:, j] = np.imag(self.psi)
        
        return self.x, self.V, self.B, self.C
    
    def calculate_transmission_reflection(self):
        """计算透射系数和反射系数"""
        # 计算透射系数和反射系数
        psi_final = self.psi_history[:, -1]
        # 假设势垒在x=110处，计算x>110区域的概率作为透射系数
        transmission_idx = self.x > self.barrier_position + self.barrier_width
        reflection_idx = self.x < self.barrier_position - self.barrier_width
        
        transmission_prob = np.sum(np.abs(psi_final[transmission_idx])**2) * self.dx
        reflection_prob = np.sum(np.abs(psi_final[reflection_idx])**2) * self.dx
        
        return transmission_prob, reflection_prob
    
    def plot_potential(self):
        """绘制势函数分布图"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.x, self.V, 'k-', linewidth=2)
        plt.xlabel('位置 x')
        plt.ylabel('势能 V(x)')
        plt.title('势能函数分布')
        plt.grid(True)
        plt.show()
    
    def plot_evolution(self):
        """绘制波函数概率密度随时间的演化"""
        plt.figure(figsize=(12, 8))
        
        # 绘制势函数（按比例缩放以便于观察）
        plt.plot(self.x, self.V / np.max(self.V) * 0.5, 'k--', label='势能（缩放）')
        
        # 绘制不同时间的概率密度
        time_points = [0, self.Nt//4, self.Nt//2, 3*self.Nt//4, self.Nt-1]
        for i, t in enumerate(time_points):
            plt.plot(self.x, np.abs(self.psi_history[:, t])**2, 
                     label=f'时间 t={t*self.dt:.2f}')
        
        plt.xlabel('位置 x')
        plt.ylabel('概率密度 |ψ(x,t)|²')
        plt.title('波函数概率密度随时间的演化')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def animate_probability_density(self, save=False, filename='quantum_tunneling.gif'):
        """创建波函数概率密度演化的动画"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 绘制势函数（按比例缩放）
        ax.plot(self.x, self.V / np.max(self.V) * 0.5, 'k--', label='势能（缩放）')
        
        # 初始化概率密度曲线
        line, = ax.plot(self.x, np.abs(self.psi_history[:, 0])**2, 'b-', linewidth=2)
        
        # 计算透射系数和反射系数
        transmission_prob, reflection_prob = self.calculate_transmission_reflection()
        
        ax.set_xlabel('位置 x')
        ax.set_ylabel('概率密度 |ψ(x,t)|²')
        ax.set_title(f'量子隧穿动画 (透射率: {transmission_prob:.4f}, 反射率: {reflection_prob:.4f})')
        ax.grid(True)
        ax.legend()
        
        # 设置y轴范围
        max_prob = np.max(np.abs(self.psi_history)**2)
        ax.set_ylim(0, max_prob * 1.1)
        
        def update(frame):
            line.set_ydata(np.abs(self.psi_history[:, frame])**2)
            ax.set_title(f'量子隧穿动画 (时间: {frame*self.dt:.2f}, '
                         f'透射率: {transmission_prob:.4f}, 反射率: {reflection_prob:.4f})')
            return line,
        
        # 创建动画
        ani = FuncAnimation(fig, update, frames=range(0, self.Nt, max(1, self.Nt//100)), 
                            interval=50, blit=True)
        
        if save:
            ani.save(filename, writer='pillow', fps=30)
        
        plt.show()
        return ani


if __name__ == "__main__":
    # 创建求解器实例
    solver = QuantumTunnelingSolver()
    
    # 绘制势函数
    solver.plot_potential()
    
    # 求解薛定谔方程
    x, V, B, C = solver.solve_schrodinger()
    
    # 计算透射系数和反射系数
    transmission, reflection = solver.calculate_transmission_reflection()
    print(f"透射系数: {transmission:.4f}")
    print(f"反射系数: {reflection:.4f}")
    print(f"总概率: {transmission + reflection:.4f}")
    
    # 绘制波函数演化
    solver.plot_evolution()
    
    # 创建动画
    solver.animate_probability_density(save=False)
