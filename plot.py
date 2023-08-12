import torch
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import matplotlib.dates as md
from matplotlib.dates import HOURLY
from matplotlib.dates import AutoDateLocator
import torch.nn.functional as F
from matplotlib.pyplot import MultipleLocator


def plot_diversity_temporal(path, loc):
    # 通过 figsize 调整图表的长宽比例，使得坐标轴上的刻度不至于太挤
    data = np.load(path)
    data = data['preds']
    a = np.array(data, np.float32)  # T*32*32

    # y = a[73:76,:,2,2,0].flatten()

    # R1[23,23] Meta邻近节点 [23,22] [23,24] [23,21],[23,25]  NoMata邻近节点 [23,20][23,19][23,26][23,27]
    # R2[14,7]  邻近节点 [14,6] [14,8] [15,6] [14,9]
    # R3[14,6]

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot()

    # 给定的日期格式可以有多种 20201001   2020-10-01  2020/10/01
    start_time_string = "20140415 01:00:00"
    end_time_string = "20140418 00:00:00"

    # 根据日期生成时间轴
    # freq 用来指明以多大间隔划分整个时间区间
    # 10T/10min 按 10 分钟划分，同理其他常见的时间跨度有
    # W 周、D 天（默认值）、H 小时、S 秒、L/ms 毫秒、U/us 微秒、N 纳秒
    # 其他具体的详见 https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    x = pd.date_range(start=start_time_string, end=end_time_string, freq="1H")

    # 二维上的点是相互对应的，根据x个数随机生成
    y1 = a[:, loc['loc1'][0], loc['loc1'][1]].flatten()
    y2 = a[:, loc['loc2'][0], loc['loc2'][1]].flatten()
    y3 = a[:, loc['loc3'][0], loc['loc3'][1]].flatten()
    y4 = a[:, loc['loc4'][0], loc['loc4'][1]].flatten()
    y5 = a[:, loc['loc5'][0], loc['loc5'][1]].flatten()

    array = pd.DataFrame({
        'y1':y1,
        'y2':y2,
        'y3':y3,
        'y4':y4,
        'y5':y5
    })
    print(array.corr())

    # x,y按离散的关系添加在图中
    ax.plot(x, y1, color="blue", linestyle="solid", label="G1")
    ax.plot(x, y2, color="orange", linestyle="dashed", label="G2")
    ax.plot(x, y3, color="green", linestyle="dotted", label="G3")
    ax.plot(x, y4, color="red", linestyle="dashdot", label="G4")
    ax.plot(x, y5, color="purple", linestyle="solid", label="G5")

    formatter = plt.FuncFormatter(time_ticks)

    # 在图中应用自定义的时间刻度
    ax.xaxis.set_major_formatter(formatter)

    # minticks 需要指出，值的大小决定了图是否能按 10min 为单位显示
    # 值越小可能只能按小时间隔显示
    locator = AutoDateLocator()
    # pandas 只生成了满足 10min 的 x 的值，而指定坐标轴以多少的时间间隔画的是下面的这行代码
    # 如果是小时，需要在上面导入相应的东东 YEARLY, MONTHLY, DAILY, HOURLY, MINUTELY, SECONDLY, MICROSECONDLY
    # 并按照下面的格式照葫芦画瓢
    locator.intervald[HOURLY] = [24]  # 10min 为间隔
    ax.xaxis.set_major_locator(locator=locator)

    # 设置xLabel与yLabel的名称
    ax.set_xlabel('Time')  # 设置x轴名称 x label
    ax.set_ylabel('Inflow')  # 设置y轴名称 y l

    # 旋转刻度坐标的字符，使得字符之间不要太拥挤
    fig.autofmt_xdate()
    plt.legend()

    plt.show()


def time_ticks(x, pos):
    # 在pandas中，按 10min 生成的时间序列与 matplotlib 要求的类型不一致
    # 需要转换成 matplotlib 支持的类型
    x = md.num2date(x)

    # 时间坐标是从坐标原点到结束一个一个标出的
    # 如果是坐标原点的那个刻度则用下面的要求标出刻度
    # if pos == 0:
    #     # %Y-%m-%d
    #     # 时间格式转换的标准是按  2020-10-01 10:10:10.0000 标记的
    #     fmt = "%Y-%m-%d %H:%M:%S.%f"
    # # 如果不是是坐标原点的那个刻度则用下面的要求标出刻度
    # else:
    #     fmt = "%Y-%m-%d.%f"
    fmt = "%Y-%m-%d.%f"
    # 根据fmt的要求画时间刻度
    label = x.strftime(fmt)

    # 当fmt有%s时需要下面的代码
    label = label.rstrip("0")
    label = label.rstrip(".")

    # 截断了秒后面的.000
    return label


def plot_bar_corr(metaArray, NoMetaArray, yMin, yMax):
    label = ('G2', 'G3', 'G4', 'G5')
    bar_width = 0.2
    bar_x = np.arange(len(label))

    fig = plt.figure(figsize=(8,4))

    ax = fig.add_subplot(111)
    bar1 = ax.bar(x=bar_x-bar_width/2, height=NoMetaArray, width=bar_width, color='#1f70a9')
    bar2 = ax.bar(x=bar_x+bar_width/2, height=metaArray, width=bar_width, color='#65a9d7')

    ax.set_xlabel('4 neighbors')
    ax.set_ylabel('CORR')
    ax.set_xticks(range(4))
    ax.set_xticklabels(label)
    plt.ylim(yMin, yMax)
    ax.legend((bar1, bar2), ('GSTA', 'MGSTA'))

    plt.show()


if __name__ == '__main__':
    MetaPath = "data/plotArray/plotMeta.npz"
    NoMetaPath = "data/plotArray/plotNoMetaH8.npz"
    # loc1
    MetaLoc1 = {'loc1':[23,23],'loc2':[23,22],'loc3':[23,21],'loc4':[23,24],'loc5':[23,25]}
    NoMetaLoc1 = {'loc1':[23,23],'loc2':[23,27],'loc3':[23,20],'loc4':[23,25],'loc5':[23,26]}
    # loc2
    MetaLoc2 = {'loc1':[18,23],'loc2':[18,22],'loc3':[18,21],'loc4':[17,23],'loc5':[17,23]}
    NoMetaLoc2 = {'loc1':[18,23],'loc2':[18,22],'loc3':[18,21],'loc4':[18,24],'loc5':[18,20]}

    plot_diversity_temporal(MetaPath,MetaLoc2)
    plot_diversity_temporal(NoMetaPath,NoMetaLoc2)

    # corr MetaLoc1 y1[y1,y2,y3,y4,y5] [1.000000  0.982637  0.963984  0.978201  0.949610]
    # corr NoMetaLoc1 y1[y1,y2,y3,y4,y5] [1.000000  0.836103  0.943847  0.769831  0.934749]

    # corr MetaLoc2 y1[y1,y2,y3,y4,y5] [1.000000  0.980987  0.806587  0.961997  0.961997]
    # corr NoMetaLoc2 y1[y1,y2,y3,y4,y5] [1.000000  0.976575  0.773202  0.958839  0.795431]

    # plot_bar_corr([0.982637, 0.963984, 0.978201, 0.949610], [0.931871, 0.943847, 0.949350, 0.934749],0.8,1.1)
    # plot_bar_corr([0.980987, 0.806587, 0.961997, 0.961997], [0.976575, 0.773202, 0.958839, 0.795431],0.6,1.05)


