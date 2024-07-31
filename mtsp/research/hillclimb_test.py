import numpy as np
import os
import csv
import copy
import glob

from mtsp.algorithms.hillclimber import stochastic_hillclimber
import mtsp.algorithms.hillclimber as hillclimber
import mtsp.algorithms.random_sample as random_sample
import mtsp.algorithms.k_means as k_means
from mtsp.classes.classes import MTSP
from mtsp.algorithms import random_sample as rd
from mtsp.research import get_instances, write_to_csv

def hillclimb_on_mulptiple_paths(paths, n_hillclimbers=1, n_steps=1000, n_couriers=None):
    for path in paths:
        mtsp = MTSP(data_path=path, data_type="euclidean", n_couriers=n_couriers)
        hillclimb_on_data(mtsp, n_hillclimbers, n_steps)

def hillclimb_on_data(mtsp, n_hillclimbers, n_steps):

    for n_h in range(n_hillclimbers):
        routeplanning, scores = hillclimber.stochastic_hillclimber(mtsp, n_steps)
        for score in scores:
            print(score)

    return

def hillclimb_for_now():
    inpath = os.path.join("datasets", "synthetic", "amsterdam", "n100", "0000")
    for n_couriers in range(1,7):
        print("weyooo", n_couriers)
        mtsp = MTSP(data_path=inpath, data_type="euclidean", n_couriers=n_couriers)

def hillclimb_and_keep_final():
    inpath = os.path.join("datasets", "synthetic", "amsterdam", "n100", "0000")

    for n_couriers in range(1,7):
        print("weyooo", n_couriers)
        mtsp = MTSP(data_path=inpath, data_type="euclidean", n_couriers=n_couriers)

        final_uniform_scores = []
        for i in range(1000):
            print("uniform", i)
            routeplanning = rd.gen_random_planning(mtsp)
            routeplanning, scores = hillclimber.stochastic_hillclimber(mtsp, 1000)
            final_uniform_scores.append([i,routeplanning.calculate_score()])
        
        final_fair_scores = []
        for i in range(1000):
            print("fair", i)
            routeplanning = rd.gen_random_fair_planning(mtsp)
            routeplanning, scores = hillclimber.stochastic_hillclimber(mtsp, 1000)
            final_fair_scores.append([i,routeplanning.calculate_score()])

        outfolder = os.path.join("results", "synthetic", "amsterdam", "n100", "0000", "euclidean", f"c{n_couriers}")
        os.makedirs(outfolder, exist_ok=True)
        
        with open(os.path.join(outfolder, "final_hc_uniform.csv"), "w", newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for score in final_uniform_scores:
                writer.writerow(score)

        with open(os.path.join(outfolder, "final_hc_fair.csv"), "w", newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for score in final_fair_scores:
                writer.writerow(score)

def hillclimb_n_times(n, iterations):
    instance_paths = get_instances.get_test_instances()
    for path in instance_paths:
        print(path)
        in_path = os.path.join("datasets", path)
        for c in range(1,7):
            out_folder = os.path.join("results", path, "euclidean", f"c{c}")
            for i in range(n):
                for step_type in ["insert", "swap", "reverse", "random"]:
                    file_name = f"hc_steps_{step_type}_{str(i).zfill(4)}.csv"
                    out_path = os.path.join(out_folder, file_name)

                    mtsp = MTSP(in_path, "euclidean", c)
                    routeplanning, scores = hillclimber.stochastic_hillclimber(mtsp, iterations, step_type=step_type)
                    write_to_csv.scores_to_csv(scores, out_path)

def compare_hillclimber_different_starts(inpath, outpath, n_couriers, n_times=[0]):
    mtsp = MTSP(data_path=inpath, data_type="euclidean", n_couriers=n_couriers)

    outfolder = os.path.join(outpath, "euclidean", f"c{n_couriers}")
    os.makedirs(outfolder, exist_ok=True)

    steps = [["insert"], ["swap"], ["reverse"], ["distribute"], ["insert", "swap", "reverse"], ["insert", "swap", "reverse", "distribute"]]
    step_names = ["insert", "swap", "reverse", "distribute", "random", "random2"]

    for i, nth_time in enumerate(n_times):
        # get starting routeplannings
        routeplanning_uniform = random_sample.gen_random_planning(mtsp)
        for j, step_type in enumerate(steps):
            if "distribute" in step_type and n_couriers == 1:
                continue
            # print(f"{i}, {step_type}")
            # Climb on uniform
            routeplanning_copy = copy.deepcopy(routeplanning_uniform)
            _, scores = hillclimber.stochastic_hillclimber(mtsp, 100000, routeplanning_copy, step_type=step_type)
            with open(os.path.join(outfolder, f"hc_{step_names[j]}_{str(nth_time).zfill(4)}.csv"), "w", newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                for score in scores:
                    writer.writerow(score)

def run_hillclimber_experiment():
    inpaths = os.path.join("datasets", "thesis", "amsterdam", "**")
    inpaths = glob.glob(inpaths, recursive=False)
    outpaths = os.path.join("results", "thesis", "amsterdam", "**")
    outpaths = glob.glob(outpaths, recursive=False)

    start_path = os.path.join("datasets", "thesis", "amsterdam", "0000")
    started = False
    
    for inpath, outpath in zip(inpaths, outpaths):
        if inpath == start_path:
            started = True

        if started:
            for n_couriers in range(1,7):
                print(inpath, n_couriers)
                compare_hillclimber_different_starts(inpath, outpath, n_couriers, n_times=[0])

def count_mixtures():
    inpaths = os.path.join("datasets", "thesis", "amsterdam", "**")
    inpaths = glob.glob(inpaths, recursive=False)
    outpaths = os.path.join("results", "thesis", "amsterdam", "**")
    outpaths = glob.glob(outpaths, recursive=False)
            
    start_path = os.path.join("datasets", "thesis", "amsterdam", "0000")
    started = False
    
    for inpath, outpath in zip(inpaths, outpaths):
        outfolder = os.path.join(outpath, "euclidean", "c3")
        if inpath == start_path:
            started = True

        if started:
            print(inpath)
            mtsp = MTSP(data_path=inpath, data_type="euclidean", n_couriers=3)
            
            for i, steps in enumerate([["insert", "swap", "reverse"], ["insert", "swap", "reverse", "distribute"]]):
                routeplanning_uniform = random_sample.gen_random_planning(mtsp)
                _, scores = hillclimber.counting_hillclimber(mtsp, 100000, routeplanning_uniform, step_type=steps)
                with open(os.path.join(outfolder, f"mix_test_{i}.csv"), "w", newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',')
                    for score in scores:
                        writer.writerow(score)
                
            