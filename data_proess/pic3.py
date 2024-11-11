import re
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict
from matplotlib.font_manager import FontProperties

def line_chart():
    font = FontProperties(family='Times New Roman', size=12)

    x = [i for i in range(11)]
    x_l = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    # 柱状图的数据
    y1 = [24.576, 25.6, 25.6, 27.328, 30.784, 48.128, 89.088, 172.032, 330.752, 653.312, 1262.59]
    y2 = [6.144, 6.528, 7.168, 7.168, 7.168, 7.168, 8.192, 12.288, 17.76, 31.744, 57.344]
    y3 = [9.216, 8.512, 8.992, 8.928, 8.352, 9.216, 11.264, 19.168, 29.696, 55.296, 102.4]

    # 折线图的数据
    t1 = [0.568847656, 1.0921875, 2.184375, 4.092505855, 7.266112266, 9.295212766, 10.04310345, 10.40178571, 10.82043344, 10.95611285, 11.072478]
    t2 = [2.363606771, 4.449142157, 8.103794643, 16.20758929, 32.41517857, 64.83035714, 113.453125, 151.2708333, 209.3261261, 234.2258065, 253.2435826]
    t3 = [2.855251736, 6.182800752, 11.70551601, 23.57885305, 50.40996169, 91.36805556, 149.5113636, 175.7195326, 226.8448276, 243.6481481, 256.9726563]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 4))

    # 共享x轴，创建第二个y轴
    ax1_t = ax1.twinx()
    ax2_t = ax2.twinx()
    ax3_t = ax3.twinx()

    # 绘制柱状图
    bars1 = ax1.bar(x, y1, color='#4A7298')
    bars2 = ax2.bar(x, y2, color='#4A7298')
    bars3 = ax3.bar(x, y3, color='#4A7298')

    # 绘制折线图
    line1, = ax1_t.plot(x, t1, '#000000', marker='o', markersize=15, markerfacecolor='#F3C846')
    line2, = ax2_t.plot(x, t2, '#000000', marker='o', markersize=15, markerfacecolor='#F3C846')
    line3, = ax3_t.plot(x, t3, '#000000', marker='o', markersize=15, markerfacecolor='#F3C846')
    line1_c = ax1.axhline(13.98, color='#B8474D', linestyle="--", linewidth=2.5)
    line2_c = ax2.axhline(14.522, color='#B8474D', linestyle="--", linewidth=2.5)
    line3_c = ax3.axhline(26.314, color='#B8474D', linestyle="--", linewidth=2.5)

    ax1.grid(axis='y', color='gray', linestyle='--', linewidth=0.5)
    ax2.grid(axis='y', color='gray', linestyle='--', linewidth=0.5)
    ax3.grid(axis='y', color='gray', linestyle='--', linewidth=0.5)

    # 设置柱状图的y轴标签
    ax1.set_ylabel('Lantency($\mu s$)', color='#000000', fontproperties=font)
    ax2.set_ylabel('Lantency($\mu s$)', color='#000000', fontproperties=font)
    ax3.set_ylabel('Lantency($\mu s$)', color='#000000', fontproperties=font)
    ax1.tick_params(axis='y', colors='#000000')
    ax2.tick_params(axis='y', colors='#000000')
    ax3.tick_params(axis='y', colors='#000000')

    # 设置折线图的y轴标签
    ax1_t.set_ylabel('Throughput speedup', color='#000000', fontproperties=font)
    ax2_t.set_ylabel('Throughput speedup', color='#000000', fontproperties=font)
    ax3_t.set_ylabel('Throughput speedup', color='#000000', fontproperties=font)
    ax1_t.tick_params(axis='y', colors='#000000')
    ax2_t.tick_params(axis='y', colors='#000000')
    ax3_t.tick_params(axis='y', colors='#000000')

    # 设置x轴标签
    ax1.set_xlabel('(a) $n=653$', fontproperties=font)
    ax2.set_xlabel('(b) $n=761$', fontproperties=font)
    ax3.set_xlabel('(c) $n=1277$', fontproperties=font)
    ax1.set_xticks(x)
    ax2.set_xticks(x)
    ax3.set_xticks(x)
    ax1.set_xticklabels([str(d) for d in x_l])
    ax2.set_xticklabels([str(d) for d in x_l])
    ax3.set_xticklabels([str(d) for d in x_l])

    # 添加标题
    # plt.title('Bar and Line Chart with Shared X-Axis')

    # 显示图例
    ax1.legend([bars1], ['Lantency'], loc='upper left', bbox_to_anchor=(0, 0.9), prop=font)
    ax2.legend([bars2], ['Lantency'], loc='upper left', bbox_to_anchor=(0, 0.9), prop=font)
    ax3.legend([bars3], ['Lantency'], loc='upper left', bbox_to_anchor=(0, 0.9), prop=font)
    legend1 = ax1_t.legend([line1], ['Throughput speedup'], loc='upper left', bbox_to_anchor=(0, 1), prop=font)
    legend2 = ax2_t.legend([line2], ['Throughput speedup'], loc='upper left', bbox_to_anchor=(0, 1), prop=font)
    legend3 = ax3_t.legend([line3], ['Throughput speedup'], loc='upper left', bbox_to_anchor=(0, 1), prop=font)
    ax1_t.add_artist(legend1)
    ax2_t.add_artist(legend2)
    ax3_t.add_artist(legend3)
    ax1_t.legend([line1_c], ['C baseline latency'], loc='upper left', bbox_to_anchor=(0, 0.8), prop=font)
    ax2_t.legend([line2_c], ['C baseline latency'], loc='upper left', bbox_to_anchor=(0, 0.8), prop=font)
    ax3_t.legend([line3_c], ['C baseline latency'], loc='upper left', bbox_to_anchor=(0, 0.8), prop=font)

    # plt.tight_layout()
    plt.subplots_adjust(left=0, right=1, bottom=0.1, top=0.9, wspace=0, hspace=0)

    # 显示图形
    plt.show()
    plt.savefig('linechat.pdf', format='pdf')

line_chart()