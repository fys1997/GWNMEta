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


def plot_diversity_temporal():
    # 通过 figsize 调整图表的长宽比例，使得坐标轴上的刻度不至于太挤
    f = h5py.File('data/BJ_FLOW.h5', 'r')
    a = np.array(f['/data'][()], np.float32)  # date hour x y 2

    # y = a[73:76,:,2,2,0].flatten()

    # R1[23,23] 邻近节点 [23,22] [23,24] [23,21] [23,25]
    # R2[14,7]  邻近节点 [14,6] [14,8] [15,6] [14,9]
    # R3[14,6]

    locX1 = 14
    locY1 = 6

    locX2 = 14
    locY2 = 5

    locX3 = 14
    locY3 = 9

    fig = plt.figure(figsize=(19, 9))
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
    y1 = a[73:76, :, locX1, locY1, 0].flatten()
    y2 = a[73:76, :, locX2, locY2, 0].flatten()
    y3 = a[73:76, :, locX3, locY3, 0].flatten()

    # x,y按离散的关系添加在图中
    ax.plot(x, y1, color="blue", linestyle="solid", label="R1")
    ax.plot(x, y2, color="green", linestyle="dashed", label="R2")
    ax.plot(x, y3, color="black", linestyle="dotted", label="R3")

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


if __name__ == '__main__':
    plot_diversity_temporal()