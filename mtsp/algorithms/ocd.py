import numpy as np
import random

import mtsp.algorithms.random_sample as rd
import mtsp.algorithms.k_means as k_means
import mtsp.algorithms.hillclimber as hillclimber

"""
Self-made algorithm that alternates between optimization, clustering, and distributing nodes. Not used in final research.
"""

def distribute_to_shorter_route(mtsp, n, routeplanning):
    routes = routeplanning.split_to_lists()

    scores = []
    for route in routes:
        scores.append(mtsp.route_length(route))

    for k in range(n):
        # Get longer route i and shorter route j
        route_tuple = random.sample(range(len(routes)), 2)
        i = route_tuple[np.argmax([scores[x] for x in route_tuple])]
        j = route_tuple[np.argmin([scores[x] for x in route_tuple])]

        # Sample node from longer route and find its added distance
        node_ix = np.random.randint(0, len(routes[i]))
        node = routes[i][node_ix]
        ad_i = mtsp.added_distance_in_route(node_ix, routes[i])

        # Find best place in shorter route and find its added distance
        place, ad_j = mtsp.best_place_in_route(node, routes[j])

        # Move node if added distance is smaller in j
        if ad_j < ad_i:
            del routes[i][node_ix]
            routes[j] = routes[j][:place] + [node] + routes[j][place:]

            # Recalibrate scores of i and j
            scores[i] = scores[i] - ad_i
            scores[j] = scores[j] + ad_j

    routeplanning.lists_to_permutation(routes)

    return routeplanning

def ocd_algorithm(mtsp, n, routeplanning=None, clustering=True, n_c=2, n_o=100, n_d=100):
    if routeplanning == None:
        routeplanning = rd.gen_random_planning()

    for i in range(n):
        print(i, routeplanning.calculate_score())
        # Cluster step
        if clustering == True and i < n/2:
            routeplanning = k_means.naive_k_means(mtsp, n_c, routeplanning)

        # Optimize step
        routeplanning = hillclimber.hillclimb_routes_seperately(mtsp, n_o, routeplanning=routeplanning, step_type=["insert", "swap", "reverse"])

        # Distribute step
        routeplanning = distribute_to_shorter_route(mtsp, n_d, routeplanning)
    
    return routeplanning

def od_algorithm(mtsp, n, routeplanning=None):
    return ocd_algorithm(mtsp, n, routeplanning=routeplanning, clustering=False)