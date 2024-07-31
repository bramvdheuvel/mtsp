import numpy as np
import matplotlib.pyplot as plt
import csv
import glob
import os


def hc_fair_uniform_histogram(path_lists, types, outpath, title):
    """
    Takes a lists of lists and plots every list as a histogram in the same figure.
    """
    score_lists = from_paths_to_score_lists(path_lists, types)

    fig, ax = plt.subplots(tight_layout=True)
    colors = ["b", "r", "y", "m"]
    labels = ["Uniform samples", "Fair samples", "Hillclimbed uniform", "Hillclimbed fair"]
    for i, dist in enumerate(score_lists):
        hist = ax.hist(dist, density=True, histtype='step', bins=20, color=colors[i], label=labels[i])

    plt.title(title)
    plt.xlabel("Score")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.savefig(outpath)

def get_random_score_lists_couriers():
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

def from_paths_to_final_scores(files):
    scores = []
    for path in files:
        with open(path, "r", newline='') as f:
            reader = csv.reader(f, delimiter=",", quotechar="|")
            for row in reader:
                pass
            scores.append(float(row[1]))
    
    return scores

def final_scores_histogram_hc(alg, n_couriers, outpath, title):
    mutations = ["insert", "swap", "reverse", "random", "distribute", "random2"]
    labels = ["move", "swap", "2-opt", "mixture 1", "distribute", "mixture 2"]
    colors = ["b", "r", "y", "b", "r", "y"]
    ls = ["solid", "solid", "solid", "dotted", "dotted", "dotted"]
    if n_couriers == 1:
        mutations = ["insert", "swap", "reverse", "random"]
        colors = ["b", "r", "y", "b"]
        labels = ["move", "swap", "2-opt", "mixture 1"]
    lists = {}
    for mutation in mutations:
        files = os.path.join("results", "thesis", "amsterdam", "**", "euclidean", f"c{n_couriers}", f"{alg}_{mutation}_0000.csv")
        files = glob.glob(files, recursive=True)
        print(mutation)
        lists[mutation] = from_paths_to_final_scores(files)

    fig, ax = plt.subplots(tight_layout=True, dpi=1200)
    for i, mutation in enumerate(mutations):
        ax.hist(lists[mutation], density=True, histtype='step', bins=20, ls=ls[i], color=colors[i], label=labels[i])
        ax.hist(lists[mutation], density=True, histtype='stepfilled', alpha=0.2, bins=20, color=colors[i])

    # plt.title(title)
    plt.xlabel("Objective Value")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.savefig(outpath)

def final_scores_histogram(alg, n_couriers, outpath):
    mutations = ["insert", "swap", "reverse", "random", "distribute", "random2"]
    labels = ["move", "swap", "2-opt", "mixture 1", "distribute", "mixture 2"]
    colors = ["b", "r", "y", "b", "r", "y"]
    ls = ["solid", "solid", "solid", "dotted", "dotted", "dotted"]
    if n_couriers == 1:
        mutations = ["insert", "swap", "reverse", "random"]
        colors = ["b", "r", "y", "b"]
        labels = ["move", "swap", "2-opt", "mixture 1"]
        ls = ["solid", "solid", "solid", "dotted"]
    if alg in ["ga", "aco"]:
        mutations.append("none")
        labels.append("none")
        colors.append("black")
        ls.append("dashed")
    lists = {}
    for mutation in mutations:
        files = os.path.join("results", "thesis", "amsterdam", "**", "euclidean", f"c{n_couriers}", f"{alg}_{mutation}_0000.csv")
        files = glob.glob(files, recursive=True)
        print(mutation)
        lists[mutation] = from_paths_to_final_scores(files)

    fig, ax = plt.subplots(tight_layout=True, dpi=200)
    for i, mutation in enumerate(mutations):
        ax.hist(lists[mutation], density=True, histtype='step', bins=20, ls=ls[i], color=colors[i], label=labels[i])
        ax.hist(lists[mutation], density=True, histtype='stepfilled', alpha=0.2, bins=20, color=colors[i])

    # plt.title(title)
    plt.xlabel("Objective Value")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.savefig(outpath)

def final_scores_histograms(alg, outpath, title):
    mutations = ["insert", "swap", "reverse", "random", "distribute", "random2", "none"]
    labels = ["move", "swap", "2-opt", "mixture 1", "distribute", "mixture 2", "none"]
    colors = ["b", "r", "y", "b", "r", "y", "black"]
    ls = ["solid", "solid", "solid", "dotted", "dotted", "dotted", "dashed"]

    fig, ax = plt.subplots(3, 2, tight_layout=True, dpi=1800)
    for i in range(6):
        n_couriers = i+1
        if n_couriers == 1:
            mutations = ["insert", "swap", "reverse", "random"]
        else:
            mutations = ["insert", "swap", "reverse", "random", "distribute", "random2"]
        
        lists = {}
        for mutation in mutations:
            files = os.path.join("results", "thesis", "amsterdam", "**", "euclidean", f"c{n_couriers}", f"{alg}_{mutation}_0000.csv")
            files = glob.glob(files, recursive=True)
            print(n_couriers, mutation)
            lists[mutation] = from_paths_to_final_scores(files)
        
        for j, mutation in enumerate(mutations):
            ax[i//2][i%2].hist(lists[mutation], density=True, histtype='step', bins=20, ls=ls[j], color=colors[j], label=labels[j])
            ax[i//2][i%2].hist(lists[mutation], density=True, histtype='stepfilled', alpha=0.2, bins=20, color=colors[j])

        plt.title(f"m = {n_couriers}")
        plt.xlabel("Objective Value")
        plt.ylabel("Probability Density")
        plt.legend()

    plt.suptitle(title)
    plt.savefig(outpath)

def all_couriers_random_histogram(outpath, title):
    # Get lists of all scores for all courier counts (1,6)
    score_lists = get_random_score_lists_couriers()

    print(len(score_lists[0]))

    # Get labels and line colors
    fig, ax = plt.subplots(tight_layout=True, dpi=1200)
    colors = ["b", "r", "y", "b", "r", "y"]
    labels = [1,2,3,4,5,6]

    # for loop histogram
    ls = "solid"
    for i, dist in enumerate(score_lists):
        if i == 3:
            ls = "dotted"
        ax.hist(dist, density=True, histtype='step', bins=50, ls=ls, color=colors[i], label=labels[i])
        ax.hist(dist, density=True, histtype='stepfilled', alpha=0.2, bins=50, color=colors[i])

    # plot
    plt.title(title)
    plt.xlabel("Objective Value")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.savefig(outpath)

def from_paths_to_score_lists(path_lists, types):
    """
    path_lists is a list of sublists, each sublists containing a number of paths. Combine the sublists, types corresponds to the same entry
    in path_lists. if its "all" add all scores to score_list, if its last only add last value.
    """
    score_lists = []
    for i, path_list in enumerate(path_lists):
        score_lists.append([])
        for path in path_list:
            new_scores = []
            with open(path, "r", newline='') as f:
                reader = csv.reader(f, delimiter=",", quotechar="|")
                for row in reader:
                    new_scores.append(int(row[1]))
        
        if types[i] == "all":
            score_lists[i] += new_scores
        elif types[i] == "last":
            score_lists[i].append(new_scores[-1])

    return score_lists
