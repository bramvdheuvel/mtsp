import numpy as np
import matplotlib.pyplot as plt
import os
import csv
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes 
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

"""
This file contains the code used to make the plots that show the objective value as function of evaluations. 
A lot is hardcoded in this file to make the plots look as they do.
The file uses the word iterations instead of evaluations, since variable names were decided on earlier in the research.
"""

def get_list_from_file(file):
    """
    Takes a csv file with rows containing [iteration, score] and returns the list [[0, score0], [i, scorei] etc]
    """
    scores = []
    with open(file, "r", newline='') as f:
        reader = csv.reader(f, delimiter=",", quotechar="|")
        for row in reader:
            scores.append([float(row[0]), float(row[1])])

    return scores


def get_values_of_files(files):
    """
    Takes a list of file paths, turns those into lists containing (i0, score0) etc.
    Then takes the lists and turns those into ((i0, [scores at i0]), i1, [scores at i1])
    where all i's in all lists are taken into account.
    """
    # Create score lists
    scores = []
    lists = []
    for file in files:
        lists.append(get_list_from_file(file))

    # Pointers are variables showing where we are in each list
    pointers = np.zeros(len(lists), dtype=int)
    iteration = 0
    done = False
    while not done:
        done = True

        scores.append([iteration, []])
        next_iterations = np.zeros(len(lists), dtype=int)

        for i, pointer in enumerate(pointers):
            scores[-1][1].append(lists[i][pointer][1])

            # As long as there still are unvisited values in any list we're not done
            if pointer + 1 < len(lists[i]):
                next_iterations[i] = lists[i][pointer + 1][0]
                done = False
            else:
                next_iterations[i] = np.iinfo(int).max
        
        iteration = min(next_iterations)
        for i, value in enumerate(next_iterations):
            if value == iteration:
                pointers[i] += 1

    return scores

def get_average_of_scores(scores):
    """
    Takes a list of form [[i0, [values0]], [i1, [values1]], ...] and returns 2 lists of forms [i0, i1, ...] and [average0, average1, ...]
    """
    iterations, averages, maxes, mines = [], [], [], []
    for i, score in scores:
        average = sum(score)/len(score)
        maxes.append(max(score))
        mines.append(min(score))
        iterations.append(i)
        averages.append(average)
    
    iterations.append(100001)
    averages.append(averages[-1])
    maxes.append(maxes[-1])
    mines.append(mines[-1])

    return iterations, averages, maxes, mines

def get_average_of_files(files):
    return get_average_of_scores(get_values_of_files(files))

def plot_averages_over_iterations(files_lists, outpath, legends=range(1000), title="wip"):
    """
    Takes multiple lists of files, calculates the average/iterations for every list, and plots them in same graph.
    """
    print(legends, legends[0], len(files_lists))
    plt.figure(dpi=1200)
    for i, files in enumerate(files_lists):
        iterations, averages, _, _ = get_average_of_files(files)
        plt.plot(iterations, averages, label=legends[i])
    plt.legend()

    plt.title(title)
    plt.xlabel("Objective Value")
    plt.ylabel("Evaluations")
    plt.legend()
    plt.savefig(outpath)

def create_average_lists(alg):
    inpath = os.path.join("results", "thesis", "amsterdam")
    mutation_types = ["insert", "swap", "reverse", "random", "distribute", "random2", "none"]
    outfolder = os.path.join("results", "thesis", "averages")
    os.makedirs(outfolder, exist_ok=True)
    for n_couriers in range(1,7):
        for mutation in mutation_types:
            if n_couriers == 1 and mutation in ["distribute", "random2"]:
                continue
            print(n_couriers, mutation)
            files = []
            for instance in range(1000):
                files.append(os.path.join(inpath, str(instance).zfill(4), "euclidean", f"c{n_couriers}", f"{alg}_{mutation}_{str(0).zfill(4)}.csv"))

            iterations, averages, maxes, mines = get_average_of_files(files)
            outpath = os.path.join(outfolder, f"c{n_couriers}_{alg}_{mutation}.csv")
            with open(outpath, "w", newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                for iteration, average, max_value, min_value in zip(iterations, averages, maxes, mines):
                    writer.writerow([iteration, average, max_value, min_value])

def read_average_files(alg):
    inpath = os.path.join("results", "thesis", "averages")
    mutation_types = ["insert", "swap", "reverse", "random", "distribute", "random2"]
    if alg in ["ga", "aco"]:
        mutation_types.append("none")
    score_lists = {}
    for n_couriers in range(1,7):
        score_lists[n_couriers] = {}
        for mutation in mutation_types:
            if n_couriers == 1 and mutation in ["distribute", "random2"]:
                continue
            file = os.path.join(inpath, f"c{n_couriers}_{alg}_{mutation}.csv")
            iterations = []
            averages = []
            maxes = []
            mines = []
            with open(file, "r", newline='') as f:
                reader = csv.reader(f, delimiter=",", quotechar="|")
                for row in reader:
                    iterations.append(float(row[0]))
                    averages.append(float(row[1]))
                    maxes.append(float(row[2]))
                    mines.append(float(row[3]))
            score_lists[n_couriers][mutation] = [iterations, averages, maxes, mines]
    
    return score_lists

def pick_zoom(alg):
    """ Hardcoded zoom amounts to make the zoomed in portion about the same size """
    if alg == "hc":
        zoomed = [0, 3.2, 5.5, 6.1, 7.2, 7.9, 8.6]
    elif alg == "sa":
        zoomed = [0, 4, 3.7, 4, 3.7, 3.7, 3.5]
    elif alg == "ga":
        zoomed = [0, 30, 17, 5.5, 3.4, 2.5, 2.2]
    elif alg == "aco":
        zoomed = [0, 18, 13.5, 10, 9, 8, 7.7]
    return zoomed

def plot_averages_one_courier_from_file(alg, n_couriers, lists, mutation_types, outpath):
    plt.figure(tight_layout=True, dpi=200)
    ax = plt.axes()
    labels = ["move", "swap", "reverse", "mixture 1", "distribute", "mixture 2", "none"]
    zoomed = pick_zoom(alg)
    colors = ["b", "r", "y", "b", "r", "y", "black"]
    ls = ["solid", "solid", "solid", "dashed", "dashed", "dashed", "dotted"]
    clists = []
    min_val = 100
    max_val = 0
    
    for i,mutation in enumerate(mutation_types):
        if n_couriers == 1 and mutation in ["distribute", "random2"]:
                continue
        clist = lists[mutation]
        clists.append(clist)
        ax.plot(clist[0], clist[1], color=colors[i], linestyle=ls[i], label=labels[i])
        val = clist[1][-1]
        if val < min_val:
            min_val = val
        if val > max_val:
            max_val = val
    
    plt.legend()

    plt.xlabel("Evaluations")
    plt.ylabel("Objective Value")

    if n_couriers == 1:
        labels = ["move", "swap", "reverse", "mixture 1", "none"]
        colors = ["b", "r", "y", "b", "black"]
        ls = ["solid", "solid", "solid", "dashed", "dotted"]

    diffo = (max_val-min_val) / 6
    y1, y2 = min_val - diffo, max_val + diffo
    x1, x2 = 100000 - 8000/zoomed[n_couriers], 100000 + 8000/zoomed[n_couriers]
    
    print(y2 - y1)
    axins = zoomed_inset_axes(ax, zoomed[n_couriers], loc=9)
    for i, clist in enumerate(clists):
        axins.plot(clist[0], clist[1], color=colors[i], linestyle=ls[i])
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    plt.xticks(visible=False)
    plt.yticks()
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    plt.draw()

    plt.savefig(outpath)


def average_files_to_figures(alg):
    score_lists = read_average_files(alg)
    mutation_types = ["insert", "swap", "reverse", "random", "distribute", "random2"]
    if alg in ["ga", "aco"]:
        mutation_types.append("none")
    outfolder = os.path.join("results", "figures", "lowres")
    for n_couriers in range(1,7):
        outpath = os.path.join(outfolder, f"{alg}_evaluations_c{n_couriers}.png")
        plot_averages_one_courier_from_file(alg, n_couriers, score_lists[n_couriers], mutation_types, outpath)

def plot_average_per_mutation_type(n_couriers, instances, files, outpath, title):
    """
    Plots all mutation types against eachother for n_points and n_couriers with given start.
    Files used are given by "instances" and "files", where instances gives the folders and files the amount of files per folder
    """
    inpath = os.path.join("results", "thesis", "amsterdam")
    mutation_types = ["insert", "swap", "reverse", "random", "distribute", "random2"]
    labels = ["move", "swap", "reverse", "mixture 1", "distribute", "mixture 2"]
    files_lists = []
    for mutation_type in mutation_types:
        files_lists.append([])
        for instance in range(instances):
            for file in range(files):
                path = os.path.join(inpath, str(instance).zfill(4), "euclidean", f"c{n_couriers}", f"hc_{mutation_type}_{str(file).zfill(4)}.csv")
                files_lists[-1].append(path)
    print(len(files_lists[1]))
    plot_averages_over_iterations(files_lists, outpath, mutation_types, title)