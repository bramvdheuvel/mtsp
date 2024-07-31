import numpy as np

from mtsp.classes.classes import Routeplanning

def gen_random_planning(mtsp):
    # Create an array containing both route-split markers [0, -1, -2, ...] and deliveries [1, n]
    routes = np.arange(-(mtsp.couriers - 1), mtsp.n_nodes)
    np.random.shuffle(routes)

    routes = routes.tolist()

    return Routeplanning(mtsp, routes)

def gen_random_fair_planning(mtsp):
    deliveries = np.arange(1, mtsp.n_nodes)
    np.random.shuffle(deliveries)

    routes_split = np.array_split(deliveries, mtsp.couriers)

    routes = []
    for i in range(mtsp.couriers):
        routes.append(-i)
        routes += routes_split[i].tolist()

    return Routeplanning(mtsp, routes)

def sample_uniform(mtsp, n):
    scores = []
    for i in range(n):
        routeplanning = gen_random_planning(mtsp)
        scores.append(routeplanning.calculate_score()[0])

    return scores

def sample_fair(mtsp, n):
    scores = []
    for i in range(n):
        routeplanning = gen_random_fair_planning(mtsp)
        scores.append(routeplanning.calculate_score()[0])

    return scores