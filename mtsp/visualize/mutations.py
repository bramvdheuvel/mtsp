import numpy as np
from collections import defaultdict, Counter
import os
import csv
import matplotlib.pyplot as plt

def make_mutations_dict(ooga):
    mutations_dict = defaultdict(list)
    inpath = os.path.join("results", "thesis", "amsterdam")
    for i in range(1000):
        file = os.path.join(inpath, str(i).zfill(4), "euclidean", "c6", f"mix_test_{ooga}.csv")
        with open(file, "r", newline='') as f:
            reader = csv.reader(f, delimiter=",", quotechar="|")
            next(reader)
            for row in reader:
                mutations_dict[int(row[0])].append(row[2])

    return mutations_dict

def make_counter_list(mutations_dict):
    counter_list = []
    total_mutations = Counter()
    mutations = ["insert", "swap", "reverse", "distribute"]
    for i in range(1,100000):
        counter_list.append([])
        for mutation in mutations_dict[i]:
            total_mutations[mutation] += 1
        for mutation in mutations:
            counter_list[i-1].append(total_mutations[mutation])

    return counter_list

def save_counter_list(counter_list, outpath):
    with open(outpath, "w", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for i in range(99999):
            writer.writerow([i+1] + counter_list[i])

def make_mutations_figure():
    mutations_dict = make_mutations_dict(1)
    counter_list = make_counter_list(mutations_dict)

    mutations = ["insert", "swap", "reverse", "distribute"]
    labels = ["move", "swap", "2-opt", "distribute"]


    plt.figure(tight_layout=True, dpi=200)
    for i, mutation in enumerate(mutations):
        line = []
        for j in range(1000):
            line.append(counter_list[j][i]/sum(counter_list[j]))
        
    

        
