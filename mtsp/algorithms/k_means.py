import numpy as np

from mtsp.algorithms.random_sample import gen_random_planning, gen_random_fair_planning
from mtsp.data_scripts.helpers import haversine

"""
Contains code for a naive k-means algorithm, not used in final research.
"""

def k_means_update(mtsp, clusters):
    centroids = []

    for cluster in clusters:
            centroid = [0,0]
            for node in cluster:
                location = mtsp.get_location(node)
                centroid[0] += location[0]
                centroid[1] += location[1]
            if len(cluster) > 0:
                centroids.append([centroid[0] / len(cluster), centroid[1] / len(cluster)])
    
    while len(centroids) < len(clusters):
        centroids.append([location[0], location[1]])

    return centroids

def k_means_assigment(mtsp, centroids):
    clusters = []
    for i in range(len(centroids)):
        clusters.append([])
    
    for node in range(1, mtsp.n_deliveries + 1):
        location = mtsp.get_location(node)
        distances = []
        for centroid in centroids:
            distances.append(haversine(location, centroid))
        clusters[np.argmin(distances)].append(node)

    return clusters


def naive_k_means(mtsp, n, routeplanning=None):
    
    if routeplanning == None:
        routeplanning = gen_random_planning(mtsp)

    clusters = routeplanning.split_to_lists()
    for i in range(n):
        # Update Step
        centroids = k_means_update(mtsp, clusters)

        # Assignment Step
        clusters = k_means_assigment(mtsp, centroids)

    routeplanning.lists_to_permutation(clusters)

    return routeplanning

    
