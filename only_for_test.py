import matplotlib.pyplot as plt

def draw_ablation():
    y = [-3.489840849906151 + 7, -6.055583702985592 + 7, -3.4844833779276714 + 7]
    x = [1, 2, 3]
    colors = ['red', 'green', 'blue']
    labels = ['High', 'Low', 'High & Low']
    bars = plt.bar(x, y, color=colors, tick_label=labels, width=0.5, bottom=-6.1)
    for bar, value in zip(bars, y):
        label_position = bar.get_height() + 0.01  # 标签位置稍高于柱子顶端
        plt.text(bar.get_x() + bar.get_width() / 2, label_position, f'{value-6}', 
                ha='center', va='bottom' if value >= 0 else 'top', color='red')
    plt.gca().set_yticks([])
    plt.ylim(-6,0)
    plt.xlabel('quality of concepts')
    plt.ylabel('rate')
    plt.title('quality of concepts vs rate')
    plt.savefig('analysis/quality_of_concepts_vs_rate.png')

def draw_pretrain():
    x = [1, 2, 3, 5, 10, 30, 100]
    y = [-0.36985745073497855, 0.1528059057473291, 0.17563401899286446, 0.31169262944109755, 0.4044842154734013, 0.2554469305703446, 0.305586654884458]
    plt.plot(x, y, marker='o')
    plt.xlabel('Sample num')
    plt.ylabel('Rate')
    plt.title('Sample num vs Rate')
    plt.savefig('analysis/sample_num_vs_rate.png')

def draw_tuning():
    x = [1, 2, 3, 5, 10, 30]
    y = [-0.36985745073497855, 0.5667115435722416, 0.6356761992706395, 0.7232023223967892, 0.6893943835866749, 0.698464191375792]
    plt.plot(x, y, marker='o')
    plt.xlabel('Sample num')
    plt.ylabel('Rate')
    plt.title('Tuning: Sample num vs Rate')
    plt.savefig('analysis/tuning_sample_num_vs_rate.png')

def draw_training_process():
    path = 'logs/10-tuning.log'
    with open(path, 'r') as f:
        lines = f.readlines()
    all_scores = []
    for line in lines:
        if 'Epoch' in line:
            all_scores.append(float(line.split(' ')[-1]))
    all_scores = all_scores[:5]
    x = list(range(0, len(all_scores)))
    y = all_scores
    plt.plot(x, y, marker='o')
    plt.xlim(-0.2, 4.2)
    plt.gca().set_xticks([0,1,2,3,4])
    plt.xlabel('Epoch')
    plt.ylabel('Rate')
    plt.title('Training Process')
    plt.savefig('analysis/training_process.png')

def draw_cross_entropy():
    # 3D 表面（颜色图）演示绘制使用冷暖色图着色的 3D 表面。通过使用 antialiased=False 使表面变得不透明。
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator
    import numpy as np

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # 构建数据
    X = np.arange(0, 1, 0.02)
    Y = np.arange(0, 1, 0.02)
    X, Y = np.meshgrid(X, Y)
    Z = X*Y + (1-X)*(1-Y)

    # 绘制曲面图
    # 绘制使用冷暖色图着色的 3D 表面。通过使用 antialiased=False 使表面变得不透明。
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # 定制z轴
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # 添加一个颜色条形图展示颜色区间
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig('analysis/cross_entropy.png')

def draw_cross_entropy_partial():
    # 3D 表面（颜色图）演示绘制使用冷暖色图着色的 3D 表面。通过使用 antialiased=False 使表面变得不透明。
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator
    import numpy as np

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # 构建数据
    X = np.arange(0, 1, 0.02)
    Y = np.arange(0, 1, 0.02)
    X, Y = np.meshgrid(X, Y)
    ZX = 2*Y - 1
    ZY = 2*X - 1
    Z = np.abs(ZX) + np.abs(ZY)

    # 绘制曲面图
    # 绘制使用冷暖色图着色的 3D 表面。通过使用 antialiased=False 使表面变得不透明。
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # 定制z轴
    ax.set_zlim(0, 2.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # 添加一个颜色条形图展示颜色区间
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig('analysis/cross_entropy_partial.png')

if __name__ == "__main__":
    draw_pretrain()