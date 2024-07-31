""" Genetic algorithm:
1. Initialize Population (style=random)
2. Determine fitness of chromosomes
3. Repeat until done:
    1. Select parents (style=tournament)
    2. Crossover and Mutation (crossover_style=)
    3. Determine fitness of children
    4. Create new population out of children/previous population (style=)

"""
import random
import copy
import numpy as np

import mtsp.algorithms.random_sample as random_sample
import mtsp.algorithms.hillclimber as hc
import mtsp.algorithms.k_means as k_means

def initialize_population(mtsp, pop_size, initialization_style):
    """
    Returns a list of length {pop_size} containing routeplanning objects
    """
    population = []
    if initialization_style == "random":
        # initialize random routplanning {pop_size} times
        for i in range(pop_size):
            population.append(random_sample.gen_random_planning(mtsp))
    elif initialization_style == "random_fair":
        for i in range(pop_size):
            population.append(random_sample.gen_random_fair_planning(mtsp))
    elif initialization_style == "k_means":
        for i in range(pop_size):
            population.append(k_means.naive_k_means(mtsp, 3))
    else:
        raise ValueError("initialization_style not available")

    return population

def calculate_fitness(solutions):
    """
    Calculates the score of every routeplanning in {solutions}. Doesn't return anything
    """
    for solution in solutions:
        solution.calculate_score()

def select_parents(population, n, selection_style, size=3):
    """
    Given {population}, returns a list of {n} parents pairs according to {selection_style}.
    """

    if selection_style == "random":
        parents = select_parents_random(population, n)
    elif selection_style == "best":
        parents = select_parents_best(population, n)
    elif selection_style == "roulette":
        parents = select_parents_roulette(population, n)
    elif selection_style == "tournament":
        parents = select_parents_tournament(population, n, size)
    elif selection_style == "rank":
        parents = select_parents_rank(population, n)
    else:
        raise ValueError("selection_style not available")
    
    return parents

def select_parents_random(population, n):
    parents = []
    for i in range(n):
        parents.append(random.sample(population, 2))

    return parents

def select_parents_best(population, n):
    """
    Takes a list of solutions {population}, and returns the {n} pairs with the lowest combined score
    """
    parents_dict = {}
    parents = []
    for i, p0 in enumerate(population):
        for j, p1 in enumerate(population):
            if i == j:
                continue
            parents_dict[p0.score + p1.score] = [p0, p1]
    
    for key, val in sorted(parents_dict.items()):
        parents.append(val)

    return parents[:n]

def select_parents_roulette(population, n):
    parents = []
    weights = [solution.score for solution in population]
    for i in range(n):
        p1, p2 = random.choices(population, weights, k=2)
        while p1 == p2:
            p1, p2 = random.choices(population, weights, k=2)
        parents.append([p1, p2])

    return parents

def select_parents_tournament(population, n, size):
    parents = []
    for i in range(n):
        p1_options = random.sample(population, size)
        p1 = sorted(p1_options, key=lambda x: x.score)[0]

        p2 = sorted(random.sample(population, size), key=lambda x: x.score)[0]
        while p1 == p2:
            p2 = sorted(random.sample(population, size), key=lambda x: x.score)[0]

        parents.append([p1, p2])

    return parents

def select_parents_rank(population, n):
    parents = []
    weights = list(range(len(population), 0, -1))
    new_pop = sorted(population, key=lambda x: x.score)
    for i in range(n):
        parents.append(random.choices(new_pop, weights, k=2))

    return parents

def do_crossovers(parent_pairs, crossover_style):
    """
    Given a list {parent_pairs}, returns a list of containing one child chromosome for every pair
    """
    children = []
    n_evals = 0
    for parent_pair in parent_pairs:
        random.shuffle(parent_pair)
        if crossover_style == "pmx":
            child, n = partially_mapped_crossover(parent_pair)
        elif crossover_style == "ox1":
            child, n = order_one_crossover(parent_pair)
        elif crossover_style == "cx":
            child, n = cycle_crossover(parent_pair)
        elif crossover_style == "scx":
            child, n = sequential_constructive_crossover(parent_pair)
        else:
            raise ValueError("Crossover style not implemented")
        children.append(child)
        n_evals += n
    return children, n_evals

def partially_mapped_crossover(parent_pair):
    """
    Partially Mapped Crossover function for genetic algorithms as in the literature. 
    """ 
    p1 = parent_pair[0].routes
    p2 = parent_pair[1].routes

    # if p1 == p2:
    #     raise ValueError("NOOO")

    # Sample two points
    point_i, point_j = random.sample(range(0, len(p1)), 2)
    # point_i = random.randrange(len(p1)) 
    # point_j = int(point_i + 0.5 * len(p1)) % len(p1)

    # Copy from point i to point j to child
    if point_i < point_j:
        point_list = list(range(point_i, point_j + 1))
    else:
        point_list = list(range(0, point_j + 1)) + list(range(point_i, len(p1)))

    # Make dict that connects all the points at loc in p1 and p2
    map_dict = {}
    for i in point_list:
        map_dict[p1[i]] = p2[i]

    # Fill in from j to i using the dict
    child_routes = []
    for i in range(len(p1)):
        if i in point_list:
            child_routes.append(p1[i])
        else:
            point = p2[i]
            while point in map_dict:
                point = map_dict[point]
            child_routes.append(point)

    return parent_pair[0].copy(child_routes), 1

def order_one_crossover(parent_pair):
    p1 = parent_pair[0].routes
    p2 = parent_pair[1].routes

    # Sample two points
    point_i, point_j = random.sample(range(0, len(p1)), 2)
    
    # Add the part of parent 1 that is between the sampled points to child
    if point_i < point_j:
        child_routes = p1[point_i:point_j]
    else:
        child_routes = p1[point_i:] + p1[:point_j]

    # Create set of seen nodes, this set keeps track of which nodes from parent 2 should not be added to child
    seen = set()
    for point in child_routes:
        seen.add(point)

    # Add nodes from parent 2 to child starting at the second point. Nodes are added in the order in which they are in parent two
    # All nodes in that are in seen, and therefore already in child, are skipped
    for i in range(len(p2)):
        point = p2[(i + point_j) % len(p2)]
        if point not in seen:
            child_routes.append(point)
            seen.add(point)

    child_routes = child_routes[len(p1)-i:] + child_routes[:len(p1)-i]

    return parent_pair[0].copy(child_routes), 1

def cycle_crossover(parent_pair):
    """
    Cycle Crossover function for genetic algorithms as in the literature. 
    """ 
    p1 = parent_pair[0].routes
    p2 = parent_pair[1].routes

    # Get Cycles
    cycles = []
    seen = []
    for i, val in enumerate(p1):
        if val in seen:
            continue

        cycle = [val]
        val = p2[i]
        while val != cycle[0]:
            cycle.append(val)
            val = p2[p1.index(val)]

        cycles.append(cycle)
        seen += cycle
    
    # Take all second cycles (second, fourth, etc) and create a list with the values
    exc = []
    for i, cycle in enumerate(cycles):
         if i % 2 == 1:
             exc += cycle

    # Create a new route taking into account the cycles         
    child_route = copy.copy(p1)
    for i, val in enumerate(p1):
        if val in exc:
            child_route[i] = p2[i]

    return parent_pair[0].copy(child_route), 1

def create_linked_dict(route):
    linked_dict = {}
    for i, node in enumerate(route):
        linked_dict[node] = [route[i-1], route[(i+1) % len(route)]]
    
    return linked_dict

def update_linked_dicts(node, linked_dicts):
    for d in linked_dicts:
        previous_node = d[node][0]
        next_node = d[node][1]
        d[previous_node][1] = next_node
        d[next_node][0] = previous_node

    return linked_dicts

def sequential_constructive_crossover(parent_pair):
    # get routes
    mtsp = parent_pair[0].mtsp
    p = [parent.routes for parent in parent_pair]
    distance_calls = 0

    # construct linked lists
    linked_dicts = [create_linked_dict(route) for route in p]

    # sample starting point
    point_i = random.sample(range(0, len(p[0])), 1)[0]

    # construct child
    node = p[0][point_i]
    child_route = [node]
    linked_dicts = update_linked_dicts(node, linked_dicts)
    for i in range(len(p[0]) - 1):
        # Next nodes are the first unvisited nodes in the routes
        next_nodes = [linked_dict[node][1] if linked_dict[node][1] != node else None for linked_dict in linked_dicts]

        # If a distance is zero
        distances = [mtsp.distance(node, next_node) if next_node != None else np.inf for next_node in next_nodes]
        distance_calls += 9

        # The chosen node to add to child is the closest of next nodes
        node = next_nodes[np.argmin(distances)]
        child_route.append(node)

        # Update linked dicts to take out visited nodes
        linked_dicts = update_linked_dicts(node, linked_dicts)

    if mtsp.check_validity(child_route) == False:
        raise ValueError("scx produced an illegal child")

    n_evals = distance_calls / mtsp.n_deliveries
    return parent_pair[0].copy(child_route), n_evals


def do_mutations(children, mutation_style=None, mutation_chance=0.05):
    """
    Given a list {solutions}, do mutation {mutation_style} with probability {p} on every solution,
    return list {solutions} (with mutations)
    """
    if mutation_style == None:
        pass
    elif mutation_style == "insert":
        for child in children:
            if random.random() < mutation_chance:
                child.routes = hc.stochastic_insert(child.routes)

    return children

def create_new_population(population, children, pop_size, population_style, elitism):
    """
    Given a list {population} containing the previous generation, and a list {children}, 
    create the new population of size {pop_size} according to {selection_sytle}
    """
    if population_style == "best":
        population = population + children
        population.sort(key=lambda x: x.score)
        return population[:pop_size]
    elif population_style == "children":
        children.sort(key=lambda x: x.score)
        population.sort(key=lambda x: x.score)
        population = population[:elitism] + children + population[elitism:]
        return population[:pop_size]
    
def local_optimization(population, mutation_style, n):
    new_pop = []
    for i, child in enumerate(population):
        before = child.calculate_score()[0]
        new_pop.append(hc.stochastic_hillclimber(child.mtsp, n, child, mutation_style)[0])
        after = new_pop[-1].calculate_score()[0]
        # print(before - after)

    return new_pop
        

def save_best_solution(population, best_solution):
    """
    Given the current {population} and previous best solution {best_solution}, get the fitness of the best solution this generation.
    If the fitness is higher, return best solution this generation, else return previous best solution.
    Also return best fitness this generation
    """
    new_pop = sorted(population, key=lambda x: x.score)
    if best_solution == None or new_pop[0].score <= best_solution.score:
        best_solution = new_pop[0]
    return best_solution, new_pop[0].score

def genetic_algorithm(mtsp, pop_size=50, n_children=50, n_evaluations=100000, mutation_chance=0.00,
                      initialization_style="random", selection_style="tournament", 
                      crossover_style="ox1", population_style="children",
                      mutation_style=["insert"], n_local=0, elitism=1):
    scores = []
    best_solution = None

    population = initialize_population(mtsp, pop_size, initialization_style)
    calculate_fitness(population)

    n = pop_size
    i = 0
    while n <= n_evaluations:
        best_solution, current_best_fitness = save_best_solution(population, best_solution)
        scores.append([i, current_best_fitness])

        parent_pairs = select_parents(population, n_children, selection_style)
        children, n_evals = do_crossovers(parent_pairs, crossover_style)
        n += n_evals

        if n_local > 0:
            children = local_optimization(children, mutation_style, n_local)
            n += n_local * n_children
        
        calculate_fitness(children)
        n += n_children

        population = create_new_population(population, children, pop_size, population_style, elitism)

        best_solution, current_best_fitness = save_best_solution(population, best_solution)
        scores.append([i, current_best_fitness])
        i+=1

    return best_solution, scores