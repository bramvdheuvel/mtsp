import numpy as np
import os
import csv
import matplotlib.pyplot as plt

def get_final_score_from_file(file):
    """
    Takes a csv file with rows containing [iteration, score] and returns the list [[0, score0], [i, scorei] etc]
    """
    with open(file, "r", newline='') as f:
        final_line = f.readlines()[-1]

    return int(final_line.split(",")[1])

def from_scores_to_ranks(scores):
    ranks = []
    for i, score in enumerate(scores):
        rank = 1
        for j in range(i):
            if score > scores[j]:
                rank += 1
            elif score < scores[j]:
                ranks[j] += 1
        
        ranks.append(rank)
    
    return ranks

def rank_mutation_methods(n_points_list, n_couriers_list, start_list, instances, files):
    results_path = os.path.join("results", "synthetic", "amsterdam")
    ranks = []
    for n_points in n_points_list:
        for n_couriers in n_couriers_list:
            for start in start_list:
                for instance in range(instances):
                    folder_path = os.path.join(results_path, f"n{n_points}", str(instance).zfill(4), "euclidean", f"c{n_couriers}")
                    for file in range(files):

                        scores = []
                        for mutation in ["insert", "swap", "reverse", "random"]:
                            file_path = os.path.join(folder_path, f"hc_{mutation}_{start}_{str(file).zfill(4)}.csv")
                            scores.append(get_final_score_from_file(file_path))

                        ranks.append(from_scores_to_ranks(scores))

    return ranks

def rank_start_methods(n_points_list, n_couriers_list, mutation_list, instances, files):
    results_path = os.path.join("results", "synthetic", "amsterdam")
    ranks = []
    for n_points in n_points_list:
        for n_couriers in n_couriers_list:
            for mutation in mutation_list:
                for instance in range(instances):
                    folder_path = os.path.join(results_path, f"n{n_points}", str(instance).zfill(4), "euclidean", f"c{n_couriers}")
                    for file in range(files):

                        scores = []
                        for start in ["uniform", "fair", "clustered"]:
                            file_path = os.path.join(folder_path, f"hc_{mutation}_{start}_{str(file).zfill(4)}.csv")
                            scores.append(get_final_score_from_file(file_path))

                        ranks.append(from_scores_to_ranks(scores))

    return ranks

def average_rank_start_methods(n_points_list, n_couriers_list, mutation_list, instances, files):
    average_ranks = np.zeros(3)
    ranks = rank_start_methods(n_points_list, n_couriers_list, mutation_list, instances, files)
    for rank in ranks:
        average_ranks += rank
    
    average_ranks = average_ranks / len(ranks)

    return average_ranks

def average_rank_mutation_methods(n_points_list, n_couriers_list, start_list, instances, files):
    average_ranks = np.zeros(4)
    ranks = rank_mutation_methods(n_points_list, n_couriers_list, start_list, instances, files)
    for rank in ranks:
        average_ranks += rank
    
    average_ranks = average_ranks / len(ranks)

    return average_ranks

def rank_mutations_on_all():
    n_points_list = [50,100,150,200,250]
    n_couriers_list = [1,2,3,4,5,6]
    start_list = ["uniform", "fair", "clustered"]
    instances = 10
    files = 10
    listo = []

    for n_points in n_points_list:
        for n_couriers in n_couriers_list:
            for start in start_list:
                ranks = average_rank_mutation_methods([n_points], [n_couriers], [start], instances, files)
                listo.append([n_points, n_couriers, start] + ranks.tolist())

    print(listo)
    
                        
def visualize_mutation_per_courier():
    n_points_list = [50,100,150,200,250]
    n_couriers_list = [1,2,3,4,5,6]
    start_list = ["uniform", "fair", "clustered"]
    mutation_list = ["insert", "swap", "reverse", "random"]
    instances = 10
    files = 10

    rank_per_couriers = []
    for n_couriers in n_couriers_list:
        rank_per_couriers.append(average_rank_mutation_methods(n_points_list, [n_couriers], start_list, instances, files))
    
    print(rank_per_couriers)
    rank_per_couriers = [*zip(*rank_per_couriers)]
    
    for i, ranks_list in enumerate(rank_per_couriers):
        plt.plot(range(1,7), ranks_list, label=mutation_list[i])
        
    plt.legend()
    plt.show()

def visualize_start_per_courier():
    n_points_list = [50,100,150,200,250]
    n_couriers_list = [1,2,3,4,5,6]
    start_list = ["uniform", "fair", "clustered"]
    mutation_list = ["insert", "swap", "reverse", "random"]
    instances = 10
    files = 10

    rank_per_start = []
    for n_couriers in n_couriers_list:
        rank_per_start.append(average_rank_start_methods(n_points_list, [n_couriers], mutation_list, instances, files))
    
    print(rank_per_start)
    rank_per_start = [*zip(*rank_per_start)]
    
    for i, ranks_list in enumerate(rank_per_start):
        plt.plot(range(1,7), ranks_list, label=start_list[i])
        
    plt.legend()
    plt.show()