import re
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict

cycle = 1 / (6 * 10**9)

def process_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    
    pattern = r"BATCH_size\s*=\s*(\d+),NUM_thread\s*=\s*(\d+),avarage:(\d+)"
    
    matches = re.findall(pattern, content)
    
    data = {
        "batch_size": [],
        "num_thread": [],
        "average": []
    }
    
    for match in matches:
        batch_size, num_thread, average = match
        if int(batch_size) >= 256:
            data["batch_size"].append(int(batch_size))
            data["num_thread"].append(int(num_thread))
            data["average"].append(int(average) * cycle)
    
    return data

def organize_data_by_batch(data):
    organized_data = defaultdict(lambda: defaultdict(list))
    
    combined_data = list(zip(data["batch_size"], data["num_thread"], data["average"]))
    
    combined_data.sort(key=lambda x: (x[0], x[1]))
    
    for batch_size, num_thread, average in combined_data:
        organized_data[batch_size][num_thread].append(average)
    
    return organized_data

def get_filename(folder_path):
    all_data = {
        "batch_size": [],
        "num_thread": [],
        "average": []
    }
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            data = process_file(file_path)
            
            all_data["batch_size"].extend(data["batch_size"])
            all_data["num_thread"].extend(data["num_thread"])
            all_data["average"].extend(data["average"])
    
    organized_data = organize_data_by_batch(all_data)
    
    y_arrays = defaultdict(list)
    for batch_size, threads in organized_data.items():
        for num_thread, values in threads.items():
            y_arrays[f'y{batch_size}_{num_thread}'] = values
    
    return y_arrays

def draw_graph(x, y_arrays, labels, name):
    plt.clf()

    for key, label in zip(sorted(y_arrays.keys()), labels):
        plt.plot(x, y_arrays[key], label=label)

    plt.title(f'{name}')
    plt.xlabel('packet loss(%)')
    plt.ylabel('handshake time(ms)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{name}.png', format='png')
    plt.show()

# Update draw_2_chart and draw_3_chart if needed...

folder_path = '.'  # Replace with the actual folder path
y_arrays = get_filename(folder_path)
x = [1, 2, 3]

for key in sorted(y_arrays.keys()):
    print(f"{key}: {y_arrays[key]}")

labels = [f'{key}' for key in sorted(y_arrays.keys())]
draw_graph(x, y_arrays, labels, 'test3')
