import numpy as np

import matplotlib.pyplot as plt

def plot_points_and_line(x, y, a, b):
    """
    绘制点和直线的函数。
    
    参数:
      x - 一维numpy数组或列表，包含点的x坐标
      y - 一维numpy数组或列表，包含点的y坐标
      a - 直线方程的斜率
      b - 直线方程的截距
    """
    # 确保点的数量相同
    if len(x) != len(y):
        raise ValueError("x和y的长度必须相等")
    
    # 绘制散点图
    plt.scatter(x, y, color='blue', label='points')

    # 使用x的范围来确定直线的绘制区间
    x_line = np.linspace(min(x), max(x), 100)
    y_line = a * x_line + b
    plt.plot(x_line, y_line, 'r-', label=f'line: y = {a}x + {b}')

    # 添加标签、图例和网格
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Points and a Line')
    plt.legend()
    plt.grid(True)
    plt.ylim(-2, 2)
    plt.show()

def plot_points_and_curve(x, y, a, b, c, d):
        """
        绘制点和曲线的函数。
        
        参数:
          x - 一维numpy数组或列表，包含点的x坐标
          y - 一维numpy数组或列表，包含点的y坐标
          a, b, c, d - 曲线函数的参数，其中曲线函数为: y = a*sin(b*x) + c*x + d
        """
        if len(x) != len(y):
            raise ValueError("x和y的长度必须相等")
        
        # 绘制散点图
        plt.scatter(x, y, color='blue', label='points')
        
        # 使用x的范围生成曲线的x坐标
        x_curve = np.linspace(min(x), max(x), 400)
        y_curve = a * np.sin(b * x_curve) + c * x_curve + d
        plt.plot(x_curve, y_curve, 'g-', label=f'curve: y = {a}sin({b}x) + {c}x + {d}')
        
        # 添加标签、图例和网格
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Points and a Curve')
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    # 示例数据
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 3, 2, 5, 4])
    a = 0.5  # 斜率
    b = 1.5  # 截距
    plot_points_and_line(x, y, a, b)
    c = 0.5  # 曲线参数c
    d = 0.5  # 曲线参数d
    plot_points_and_curve(x, y, a, b, c, d)