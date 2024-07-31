import numpy as np
import matplotlib.pyplot as plt
import csv
import glob
import os
import shutil

def get_data_from_list(score_list):
    std = np.std(score_list, dtype=np.float64)
    max_val = max(score_list)
    min_val = min(score_list)
    avg = sum(score_list) / len(score_list)

    return avg, std, min_val, max_val

def files_to_list(filenames):
    score_list = []
    for filename in filenames:
        with open(filename, "r", newline='') as f:
            reader = csv.reader(f, delimiter=",", quotechar="|")
            for row in reader:
                pass
            score_list.append(float(row[1]))
    
    return score_list

def get_random_data():
    score_lists = []
    for i, n_couriers in enumerate([1,2,3,4,5,6]):
        score_lists.append([])
        inpaths = os.path.join("results", "thesis", "amsterdam", "**", "euclidean", f"c{n_couriers}", "random_uniform.csv")
        inpaths = glob.glob(inpaths, recursive=True)
        print(len(inpaths))
        for path in inpaths:
            with open(path, "r", newline='') as f:
                reader = csv.reader(f, delimiter=",", quotechar="|")
                for row in reader:
                    score_lists[i].append(float(row[1]))

    return score_lists

def make_results_table(out_file):
    in_path = os.path.join("results", "thesis", "amsterdam")

    algs = ["aco"]
    opts = ["insert", "swap", "reverse", "random", "distribute", "random2", "none"]
    n_couriers = [1,2,3,4,5,6]
    table = []
    for alg in algs:
        for opt in opts:
            for m in n_couriers:
                if m == 1 and opt in ["distribute", "random2"]:
                    continue
                if alg in ["hc", "sa"] and opt == "none":
                    continue
                files = os.path.join(in_path, "**", "euclidean", f"c{m}", f"{alg}_{opt}_0000.csv")
                files = glob.glob(files, recursive=True)
                score_list = files_to_list(files)
                data = get_data_from_list(score_list)
                table.append([alg, opt, m] + list(data))

    with open(out_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|')
        for row in table:
            writer.writerow(row)