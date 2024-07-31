import os

from mtsp.research import hillclimb_test
from mtsp.research import annealing_test
from mtsp.research import ga_test
from mtsp.research import aco_test
from mtsp.data_scripts import make_datasets
from mtsp.visualize import visualize_solution, distribution

def create_datasets():
    make_datasets.make_datasets_thesis()

def run_experiments():
    hillclimb_test.run_hillclimber_experiment()
    annealing_test.run_annealing_experiment()

def make_plots():
    outpath = os.path.join("results", "figures", "test")
    os.makedirs(outpath, exist_ok=True)
    for alg in ["hc", "sa", "ga", "aco"]:
        for m in range(1,7):
            print(m)
            distribution.final_scores_histogram(alg, m, os.path.join(outpath, f"{alg}_histograms_c{m}.png"))

if __name__ == "__main__":
    # make_datasets()
    # run_experiments()
    # make_plots
    pass