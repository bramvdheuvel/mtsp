import numpy as np
import os
import csv
import copy
import glob

from mtsp.algorithms.hillclimber import simulated_annealing_additive
import mtsp.algorithms.hillclimber as hillclimber
import mtsp.algorithms.random_sample as random_sample
import mtsp.algorithms.k_means as k_means
from mtsp.classes.classes import MTSP
from mtsp.algorithms import random_sample as rd
from mtsp.research import get_instances, write_to_csv

def compare_annealing_different_starts(inpath, outpath, n_couriers, n_times=[0]):
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
            _, scores = hillclimber.simulated_annealing_additive(mtsp, 100000, cooling_scheme="quadratic", t_max=1, t_min=0.00000001, step_type=step_type)
            with open(os.path.join(outfolder, f"sa_{step_names[j]}_{str(nth_time).zfill(4)}.csv"), "w", newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                for score in scores:
                    writer.writerow(score)

def run_annealing_experiment():
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
                compare_annealing_different_starts(inpath, outpath, n_couriers, n_times=[0])
            