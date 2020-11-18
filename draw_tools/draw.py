import matplotlib.pyplot as plt


def one_d_dot_graph(data_list, low, high):
    y = [0]*len(data_list)
    plt.scatter(data_list, y, edgecolors='red')
    plt.xlim(low, high)
    plt.ylim(-1, 1)
    plt.show()


def dis_graph(key, value):
    plt.bar(x=key, height=value, color='steelblue', alpha=0.8)
    # 在柱状图上显示具体数值, ha参数控制水平对齐方式, va控制垂直对齐方式
    for xx, yy in zip(key, value):
        plt.text(xx, yy + 1, str(yy), ha='center', va='bottom', fontsize=20,
                 rotation=0)
    # 显示图例
    plt.legend()
    plt.show()