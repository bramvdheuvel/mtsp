import numpy as np
import random

import mtsp.classes.classes as classes
import mtsp.algorithms.hillclimber as hillclimber

def update_pheromones(pheromone_matrix, ants, evaporation_rate=0.1, Q=1, temp_min=None, temp_max=None):
    """ Updates the pheromone matrix given parameters and the set of ants used for updates """

    # Evaporate old pheromone
    pheromone_matrix = pheromone_matrix * (1 - evaporation_rate)

    # Add new pheromone
    for ant in ants:
        score = ant.score
        routes = ant.routes
        for i in range(len(routes)):
            pheromone_matrix[routes[i-1], routes[i]] += Q/score

    if temp_min != None or temp_max != None:
        pheromone_matrix = np.clip(pheromone_matrix, temp_min, temp_max)    

    return pheromone_matrix

def construct_ant(mtsp, pheromone_matrix, alpha, beta):
    """ Stochastic greedy algorithm constructing a solution given the mtsp (containing distance matrix) and a pheromone matrix """
    routes = []
    unvisited = np.arange(-(mtsp.couriers - 1), mtsp.n_nodes).tolist()
    distance_calls = 0

    routes.append(random.choice(unvisited))
    unvisited.remove(routes[0])

    while len(unvisited) > 0:
        current_node = routes[-1]
        weights = []

        only_hubs = True
        for node in unvisited:
            distance = mtsp.distance(current_node, node)
            distance_calls += 1
            if distance == 0:
                weight = 0
            else:
                only_hubs = False
                weight = (pheromone_matrix[current_node,node] ** alpha) * ((1/distance) ** beta)
            weights.append(weight)
        
        if only_hubs:
            weights = [1 for weight in weights]

        # print(only_hubs, weights)
        next_node = random.choices(unvisited, weights=weights, k=1)[0]

        routes.append(next_node)
        unvisited.remove(next_node)

    n_evaluations = distance_calls / mtsp.n_deliveries
    
    return classes.Routeplanning(mtsp, routes=routes), n_evaluations

def construct_ant_fair(mtsp, pheromone_matrix, alpha, beta):
    """ 
    Stochastic greedy algorithm constructing a solution given the mtsp (containing distance matrix) and a pheromone matrix. 
    Keeps sub tour lengths the same by only adding nodes to shortest sub tour
    """
    routes = [[] for _ in range(mtsp.couriers)]
    lengths = np.zeros(shape=mtsp.couriers)
    unvisited = np.arange(1, mtsp.n_nodes).tolist()
    distance_calls = 0

    while len(unvisited) > 0:
        idx = np.argmin(lengths)

        if len(routes[idx]) == 0:
            current_node = 0
        else:
            current_node = routes[idx][-1]

        weights = []

        for node in unvisited:
            distance = mtsp.distance(current_node, node)
            distance_calls += 1
            if distance == 0:
                weight = 0
            else:
                weight = (pheromone_matrix[current_node,node] ** alpha) * ((1/distance) ** beta)
            weights.append(weight)

        next_node = random.choices(unvisited, weights=weights, k=1)[0]
        routes[idx].append(next_node)
        lengths[idx] += mtsp.distance(current_node, next_node)
        unvisited.remove(next_node)

    routeplanning = classes.Routeplanning(mtsp)
    routeplanning.lists_to_permutation(routes)

    n_evaluations = distance_calls / mtsp.n_deliveries

    return routeplanning, n_evaluations

def ant_system(mtsp, n, initial_pheromones, alpha, beta, pop_size, evaporation_rate, Q, local_search=80, mutation=["insert", "swap", "reverse", "distribute"]):
    """ The ant system algorithm as introduced by Dorigo et al in the 90s, using a heuristic to keep route lengths similar """
    pheromone_matrix = np.full(shape=mtsp.distance_matrix.shape, fill_value=initial_pheromones)  
    scores = []

    best_score = np.inf
    n_evaluations = 0
    i = 0
    while n_evaluations <= n:
        ants = []
        for j in range(pop_size):
            # Construct ants
            ant, ant_evals = construct_ant_fair(mtsp, pheromone_matrix, alpha, beta)
            n_evaluations += ant_evals
            ants.append(ant)     

        # Optional local search
        if local_search > 0:
            for k, ant in enumerate(ants):
                ants[k] = hillclimber.stochastic_hillclimber(mtsp, local_search, ant, step_type=mutation)[0]
                n_evaluations += local_search

        # Update pheromones
        scores = [ant.calculate_score()[0] for ant in ants]
        pheromone_matrix = update_pheromones(pheromone_matrix, ants, evaporation_rate, Q)

        # Tracking
        current_score = np.min(scores)
        if current_score < best_score:
            best_ant = ants[np.argmin(scores)]
            best_score = current_score
        scores.append((i, max([ant.score for ant in ants])))
        print(i, current_score)
        i += 1
    
    return best_ant, scores