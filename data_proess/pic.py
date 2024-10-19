import re
import matplotlib.pyplot as plt
import numpy as np

import os
import re
import os
import re
from collections import defaultdict

cycle = 1 / (6 * 10**9) * 10**6
def process_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    # Regular expression to match the required values
    pattern = r"BATCH_size\s*=\s*(\d+),NUM_thread\s*=\s*(\d+)avarage:(\d+)"
    
    matches = re.findall(pattern, content)
    
    data = {
        "batch_size": [],
        "num_thread": [],
        "average": []
    }
    
    for match in matches:
        batch_size, num_thread, average = match
        #if(int(batch_size) >= 256):
        data["batch_size"].append(int(batch_size))
        data["num_thread"].append(int(num_thread))
        data["average"].append(int(average) * cycle)
    
    return data

def organize_data_by_thread(data):
    organized_data = defaultdict(list)
    
    combined_data = list(zip(data["batch_size"], data["num_thread"], data["average"]))
    
    # Sort combined data by num_thread and then by batch_size
    combined_data.sort(key=lambda x: (x[1], x[0]))
    
    for batch_size, num_thread, average in combined_data:
        organized_data[num_thread].append((batch_size, average))
    
    # Sort each thread's data by batch_size
    for num_thread in organized_data:
        organized_data[num_thread].sort(key=lambda x: x[0])
    
    return organized_data


def organize_data_by_batch(data):
    organized_data = defaultdict(list)
    
    combined_data = list(zip(data["batch_size"], data["num_thread"], data["average"]))
    
    # Sort combined data by batch_size and then by num_thread
    combined_data.sort(key=lambda x: (x[0], x[1]))
    
    for batch_size, num_thread, average in combined_data:
        organized_data[batch_size].append((num_thread, average))
    
    # Sort each batch_size's data by num_thread
    for batch_size in organized_data:
        organized_data[batch_size].sort(key=lambda x: x[0])
    
    return organized_data

def get_filename(folder_path):
    all_data = {
        "batch_size": [],
        "num_thread": [],
        "average": []
    }
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):  # Adjust this if your files have a different extension
            file_path = os.path.join(folder_path, filename)
            data = process_file(file_path)
            
            all_data["batch_size"].extend(data["batch_size"])
            all_data["num_thread"].extend(data["num_thread"])
            all_data["average"].extend(data["average"])
    
    organized_data = organize_data_by_thread(all_data)
    
    # Extract averages to arrays y1 to y1010
    y_arrays = defaultdict(list)
    for num_thread, values in organized_data.items():
        averages = [v[1] for v in values]
        y_arrays[f'y{num_thread}'] = averages
    
    return y_arrays

    

def draw_2_chart():


    group_names = ['1','2','4','8','16','32','64','128','256']
    values = [36.06,108.29,18.03,54.32,9.11,27.16,4.64,13.68,2.49,7.58,1.63,5.90,1.46,5.58,1.41,5.49,1.31,4.92]
    # 每个柱子的颜色
    colors = ['#D2D6F5', '#CEDBD2']#//, '#AED4E5', '#F9e9a4']

    # 每组有四个柱子
    num_groups = len(group_names)
    num_bars_per_group = 2

    # 设置图标
    icons = ['barret', 'single'], #'montgomery', 'dual']

    # 创建一个新的图和轴
    fig, ax = plt.subplots(figsize=(8, 6))

    # 设置柱子的宽度
    bar_width = 0.2

    # 计算每个组的柱子的x位置
    indices = np.arange(num_groups)

    # 绘制每组中的每个柱子
    for i in range(num_bars_per_group):
        bar_positions = indices + i * bar_width
        bars = ax.bar(bar_positions, values[i::num_bars_per_group], bar_width, color=colors[i], label=f'{icons[i]}')
        
        # 在每个柱子上方显示其值
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.0f}', va='bottom', ha='center')  # va='bottom' 在柱子顶部显示

    # 设置x轴的刻度为每组的名字
    ax.set_xticks(indices + bar_width * (num_bars_per_group - 1) / 2)
    ax.set_xticklabels(group_names)


    # 设置图例并将其放在右侧
    ax.legend(loc='upper right')

    # 设置x轴和y轴标签
    ax.set_xlabel('moduli')
    ax.set_ylabel('Total time for running 10000 times (cycles)')

    ax.set_ylim(top=180000) 
    # 设置标题
    #ax.set_title('Performance comparison of modular reduction algorithms for different pseudo-mersenne numbers')

    ax.grid(True, linewidth=0.5, linestyle=':') 
    # 调整图表以适应图例
    #plt.tight_layout(rect=[0, 0, 0.95, 1])

    # 保存并显示图表
    plt.savefig("complex.pdf")
    plt.show()



def draw_3_chart():
    # Group names
    group_names = ['1', '2', '4', '8', '16', '32', '64', '128', '256']
    
    # Values to be plotted
    values = [36.06, 108.29, 18.03, 54.32, 9.11, 27.16, 4.64, 13.68, 2.49, 7.58, 1.63, 5.90, 1.46, 5.58, 1.41, 5.49, 1.31, 4.92]
    
    # Colors for each set of bars
    colors = ['#D2D6F5', '#CEDBD2']

    # Number of groups and bars per group
    num_groups = len(group_names)
    num_bars_per_group = 2

    # Labels for each bar
    labels = ['version1', 'version2']

    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))

    # Width of each bar
    bar_width = 0.3

    # Calculate x positions for each group
    indices = np.arange(num_groups)

    # Draw bars
    for i in range(num_bars_per_group):
        bar_positions = indices + i * bar_width
        bars = ax.bar(bar_positions, values[i::num_bars_per_group], bar_width, color=colors[i], label=labels[i])
        
        # Display values on top of the bars
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', va='bottom', ha='center')

    # Set x-axis ticks and labels
    ax.set_xticks(indices + bar_width * (num_bars_per_group - 1) / 2)
    ax.set_xticklabels(group_names)

    # Set legend location
    ax.legend(loc='upper right')

    # Set x and y labels
    ax.set_xlabel('batch')
    ax.set_ylabel('avarage time(us)')

    # Optionally set y-axis limit
    ax.set_ylim(top=120) 

    # Add grid
    ax.grid(True, linewidth=0.5, linestyle=':')

    # Save and show the plot
    plt.savefig("complex.png")
    plt.show()


def draw_graph(x,y_arrays,line1_label,line2_label,line3_label,line4_label,line5_label,name):

    plt.clf()

    # 绘制第一条直线
    plt.plot(x, y_arrays['y1'], label=line1_label, color='blue', linestyle='-', marker='o')

    # 绘制第二条直线
    plt.plot(x, y_arrays['y2'], label=line2_label, color='red', linestyle='--', marker='s')

    plt.plot(x, y_arrays['y3'], label=line3_label, color='yellow', linestyle='--', marker='s')

    plt.plot(x, y_arrays['y4'], label=line4_label, color='green', linestyle='--', marker='s')

    plt.plot(x, y_arrays['y5'], label=line5_label, color='black', linestyle='--', marker='s')

    plt.plot(x, y_arrays['y6'], label=line1_label, color='blue', linestyle='-', marker='o')

    # 绘制第二条直线
    plt.plot(x, y_arrays['y7'], label=line2_label, color='red', linestyle='--', marker='s')

    plt.plot(x, y_arrays['y8'], label=line3_label, color='yellow', linestyle='--', marker='s')

    plt.plot(x, y_arrays['y9'], label=line4_label, color='green', linestyle='--', marker='s')

    plt.plot(x, y_arrays['y10'], label=line5_label, color='black', linestyle='--', marker='s')


    # 设置图表的标题和坐标轴标签
    plt.title(f'mae')
    plt.xlabel('packet loss(%)')
    plt.ylabel('handshake time(ms)')

    # 显示图例
    plt.legend()

    # 显示网格（可选）
    plt.grid(True)
    plt.savefig(f'test3.png', format='png')

    # 显示图表
    plt.show()


def draw_graph_batch(x,y_arrays,line1_label,line2_label,line3_label,line4_label,line5_label,name):

    plt.clf()

    # 绘制第一条直线
    #plt.plot(x, y_arrays['y1'], label=line1_label, color='blue', linestyle='-', marker='o')

    # 绘制第二条直线
    plt.plot(x, y_arrays['y128'], label="batch 128", color='#1E4C9C', linestyle='-')

    plt.plot(x, y_arrays['y256'], label="batch 256", color='#345D82', linestyle='-')
    
    plt.plot(x, y_arrays['y512'], label="batch 512", color='#3371B3', linestyle='-')

    plt.plot(x, y_arrays['y1024'], label="batch 1024", color='#5795C7', linestyle='-')

    plt.xticks(range(min(x), max(x) + 1, 1))



    # 设置图表的标题和坐标轴标签
    #plt.title(f'')
    plt.xlabel('number of cuda stream')
    plt.ylabel('avarage time of keygen(us)')

    # 显示图例
    plt.legend()
    plt.grid(True, linestyle='--')

    # 显示网格（可选）
    #plt.grid(True)
    plt.savefig(f'test_bat.png', format='png')

    # 显示图表
    plt.show()


def draw_compact_bar_chart(data, name):
    plt.clf()

    # 设置x轴刻度
    x = range(1, len(data) + 1)

    first_value = (float)(data[0])
    multiples = [first_value / value * 100 for value in data]

    # 绘制柱状图
    plt.bar(x, data, color='#AED4E5', width=0.7)

    plt.plot(x, multiples, color='#5795c7', linestyle='-', label='Multiples')

    # 绘制连接顶点的直线
    #plt.plot(x, data, color='#81B5D5', linestyle='--')

    # 计算并显示比率
    first_value = data[0]
    for i, value in enumerate(data):
        ratio = first_value / value
        plt.text(x[i], value + 10, f'{ratio:.2f}x', ha='center', fontsize=10)

    # 设置图表的标题和坐标轴标签
    plt.xlabel('number of cuda stream')
    plt.ylabel('average time of keygen (us)')
    #plt.title(name)

    # 设置x轴刻度间隔为1
    plt.xticks(x)

    # 显示网格为虚线
    plt.grid(True, linestyle='--', axis='y')

    # 保存图表
    plt.savefig(f'{name}.png', format='png')

    # 显示图表
    plt.show()



def draw_compact_bar_chart_with_multiple(data, name):
    plt.clf()

    # 设置x轴刻度
    x = range(1, len(data) + 1)

    # 计算倍数
    first_value = data[0]
    multiples = [first_value / value * 100 for value in data]

    # 绘制柱状图
    plt.bar(x, data, color='#c1dbd2', width=0.7)

    # 绘制倍数的折线
    plt.plot(x, multiples, color='#5795c7', linestyle='-', label='倍数')

    # 显示倍数值
    for i, mult in enumerate(multiples):
        plt.text(x[i], data[i] + 10, f'{mult:.2f}', ha='center', va='bottom', fontsize=8, color='blue')

    # 设置图表的标题和坐标轴标签
    plt.xlabel('number of cuda stream')
    plt.ylabel('average time of keygen (us)')
    plt.title(name)

    # 设置x轴刻度间隔为1
    plt.xticks(x)

    # 显示图例
    plt.legend()

    # 显示网格为虚线
    plt.grid(True, linestyle='--', axis='y')

    # 保存图表
    plt.savefig(f'{name}.png', format='png')

    # 显示图表
    plt.show()


#draw_3_chart()
folder_path = '.'  # Replace with the actual folder path
y_arrays = get_filename(folder_path)
x=[1,2,3,4,5,6,7,8,9,10]#,11]
# Print the arrays y1 to y1010
for key in sorted(y_arrays.keys(), key=lambda x: int(x[1:])):
    print(f"{key}: {y_arrays[key]}")

#print(y_arrays['y10'])

#draw_graph_batch(x,y_arrays,'1','2','3','4','5','name')

#draw_compact_bar_chart(y_arrays['y1'], "1batch")

#draw_compact_bar_chart_with_multiple(y_arrays['y1'], "1batch2")