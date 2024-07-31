import numpy as np
import os
import csv
import matplotlib.pyplot as plt

def show_scores_per_evaluation(scores):
    evaluations = []
    objective = []
    for i, score in scores:
        evaluations.append(i)
        objective.append(score)

    plt.figure(dpi=200)

    plt.plot(evaluations, objective)
    plt.show()

def save_scores_per_evaluation(scores, outpath):
    evaluations = []
    objective = []
    for i, score in scores:
        evaluations.append(i)
        objective.append(score)

    plt.figure(dpi=200)

    plt.xlabel("Evaluations")
    plt.ylabel("Objective Value")

    plt.plot(evaluations, objective)
    plt.savefig(outpath)