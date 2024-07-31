import random
import numpy as np
import copy
from collections import Counter

from mtsp.algorithms.random_sample import gen_random_planning, gen_random_fair_planning
from mtsp.classes.classes import Routeplanning
from mtsp.data_scripts.helpers import haversine

def select_step_type(step_type):
    steps = []
    for step in step_type:
        if step == "insert":
            steps.append(stochastic_insert)
        elif step == "swap":
            steps.append(stochastic_swap)
        elif step == "reverse":
            steps.append(stochastic_reverse)
        elif step == "distribute":
            steps.append(distribute_to_shorter_route)
        elif step == "cluster":
            steps.append(cluster)
        elif step == "swap_sections":
            steps.append(swap_sections2)

    if len(steps) == 0:
        raise ValueError("step_type needs to contain valid steps")
    return steps

def do_step(routeplanning, steps):
    return random.choice(steps)(routeplanning)

def stochastic_insert(routeplanning):
    """ move operator """
    routes = routeplanning.routes
    i, j = random.sample(range(0, len(routes)), 2)

    if i > j:
        routes = routes[:j] + [routes[i]] + routes[j:i] + routes[i+1:]
        # routes = routes[:j] + [i] + routes[j:i] + routes[i+1:]
    if i < j:
        routes = routes[:i] + routes[i+1:j] + [routes[i]] + routes[j:]
        # routes = routes[:i] + routes[i+1:j] + [i] + routes[j:]

    routeplanning.routes = routes
    score = routeplanning.calculate_score()

    return routeplanning, score

def stochastic_swap(routeplanning):
    """ swap operator """
    routes = routeplanning.routes
    i, j = random.sample(range(0, len(routes)), 2)

    if j < i:
        i, j = j, i

    routes = routes[:i] + [routes[j]] + routes[i+1:j] + [routes[i]] + routes[j+1:]

    routeplanning.routes = routes
    score = routeplanning.calculate_score()

    return routeplanning, score

def stochastic_reverse(routeplanning):
    """ 2-opt operator """
    routes = routeplanning.routes
    i, j = random.sample(range(0, len(routes)), 2)

    if j < i:
        i, j = j, i

    subroute = routes[i:j+1]
    routes = routes[:i] + subroute[::-1] + routes[j+1:]

    routeplanning.routes = routes
    score = routeplanning.calculate_score()

    return routeplanning, score

def random_step(routeplanning):
    """ Do one of three random mutation operators on routeplanning """
    steps = [stochastic_insert, stochastic_reverse, stochastic_swap]
    return random.choice(steps)(routeplanning)

def swap_sections(routeplanning):
    """
    Swaps 2 sections in the routeplanning, emulating an n-opt
    """
    mtsp = routeplanning.mtsp
    routes = routeplanning.routes

    points = random.sample(range(0, len(routes)), 4)
    points.sort()

    section0 = routes[points[0]:points[1]]
    if random.random() < 0.5:
        section0 = section0[::-1]
    section1 = routes[points[2]:points[3]]
    if random.random() < 0.5:
        section1 = section1[::-1]

    routes = routes[:points[0]] + section1 + routes[points[1]:points[2]] + section0 + routes[points[3]:]

    routeplanning.routes = routes
    score = routeplanning.calculate_score()

    return routeplanning, score

def swap_sections2(routeplanning):
    """
    Swaps 2 sections in the routeplanning, emulating an n-opt
    """
    mtsp = routeplanning.mtsp
    routes = routeplanning.split_to_lists()
    i, j = random.sample(range(0, len(routes)), 2)

    sections = {}
    points = {}
    for x in [i,j]:
        points[x] = sorted(random.sample(range(0, len(routes[x])), 2))
        section = routes[x][points[x][0]:points[x][1]]
        if random.random() < 0.5:
            section = section[::-1]
        sections[x] = section

    routes[i] = routes[i][:points[i][0]] + sections[j] + routes[i][points[i][1]:]
    routes[j] = routes[j][:points[j][0]] + sections[i] + routes[j][points[j][1]:]

    routeplanning.lists_to_permutation(routes)
    score = routeplanning.calculate_score()

    return routeplanning, score

def distribute_to_shorter_route(routeplanning):
    """
    Takes a random node and moves it to the spot in the permutation that minimizes total route length
    """
    mtsp = routeplanning.mtsp
    routes = routeplanning.split_to_lists()

    scores = []
    for route in routes:
        scores.append(mtsp.route_length(route))

    # Get longer route i and shorter route j
    route_tuple = random.sample(range(len(routes)), 2)
    while len(routes[route_tuple[0]]) == 0:
        route_tuple = random.sample(range(len(routes)), 2)
    # i = route_tuple[np.argmax([scores[x] for x in route_tuple])]
    # j = route_tuple[np.argmin([scores[x] for x in route_tuple])]
    i = route_tuple[0]
    j = route_tuple[1]

    # Sample node from longer route and find its added distance
    node_ix = np.random.randint(0, len(routes[i]))
    node = routes[i][node_ix]
    ad_i = mtsp.added_distance_in_route(node_ix, routes[i])

    # Find best place in shorter route and find its added distance
    place, ad_j = mtsp.best_place_in_route(node, routes[j])

    # Move node if added distance is smaller in j
    # if ad_j < ad_i:
    del routes[i][node_ix]
    routes[j] = routes[j][:place] + [node] + routes[j][place:]

    # Recalibrate scores of i and j
    scores[i] = scores[i] - ad_i
    scores[j] = scores[j] + ad_j

    routeplanning.lists_to_permutation(routes, update_score=True)

    return routeplanning, [max(scores), sum(scores)]

def distribute(mtsp, routes):
    """
    Takes a random node and moves it to the spot in the permutation that minimizes total route length
    """
    i = random.randrange(len(routes))
    while routes[i] <= 0:
        i = random.randrange(len(routes))

    node = routes[i]
    routes = routes[:i] + routes[i+1:]

    max_distance = np.inf

    for j in range(len(routes)):
        added_distance = mtsp.added_distance(routes[j-1], node, routes[j])
        if added_distance < max_distance:
            max_distance = added_distance
            distribute_spot = j

    routes = routes[:distribute_spot] + [node] + routes[distribute_spot:]

    return routes

def cluster(routeplanning):
    """ Cluster mutation operator, takes a node and puts it in the sub tour in which it is closest to the euclidean center """
    mtsp = routeplanning.mtsp

    node = np.random.choice(routeplanning.routes)
    while node <= 0:
        node = np.random.choice(routeplanning.routes)
    loc = mtsp.get_location(node)

    routeplanning.routes.remove(node)

    routes = routeplanning.split_to_lists()

    centers = []
    for route in routes:
        centers.append(mtsp.route_center(route))

    i = np.argmin(haversine(c, loc) for c in centers)
    place, _ = mtsp.best_place_in_route(node, routes[i])   

    routes[i] = routes[i][:place] + [node] + routes[i][place:]

    routeplanning.lists_to_permutation(routes)
    score = routeplanning.calculate_score()

    return routeplanning, score
    
        
def acceptance_function(temperature, old_score, new_score):
    """ Acceptance function used in simulated annealing """
    if new_score <= old_score:
        return True
    else:
        r = random.random()
        if r < np.exp((old_score - new_score)/temperature):
            return True
        else:
            return False

def linear_additive_cooling(t_max, t_min, n, i):
    return t_min + (t_max - t_min) * (n - i) / n

def exponential_additive_cooling(t_max, t_min, n, i):
    return t_min + (t_max - t_min) * 1 / (1 + np.exp((2 * np.log(t_max - t_min))/n * (i - 0.5 * n)))

def quadratic_additive_cooling(t_max, t_min, n, i):
    return t_min + (t_max - t_min) * ((n-i)/n)**2

def linear_multiplicative_cooling(temperature, cooling_parameter):
    return temperature - cooling_parameter

def exponential_multiplicative_cooling():
    pass

def simulated_annealing_additive(mtsp, n, routeplanning=None, t_max=1, t_min=0.00000001, cooling_scheme="linear", step_type=["insert"]):
    """ 
    Main additive simulated annealing loop for n evaluations, step_type is a list containing all mutation operators used.
    cooling scheme can be either linear, quadratic, or exponential
    """

     # Select hillclimbing step
    steps= select_step_type(step_type)

    if cooling_scheme == "linear":
        cooling_function = linear_additive_cooling
    elif cooling_scheme == "exponential":
        cooling_function = exponential_additive_cooling
    elif cooling_scheme == "quadratic":
        cooling_function = quadratic_additive_cooling

    scores = []
    if routeplanning == None:
        routeplanning = gen_random_planning(mtsp)
    
    score = routeplanning.calculate_score()[0]
    scores.append([0,score])
    for i in range(n):
        t = cooling_function(t_max, t_min, n, i)
        new_planning, new_score = do_step(routeplanning.copy(), steps)

        if acceptance_function(t, score, new_score[0]):
            routeplanning = new_planning
            score = new_score[0]
            scores.append([i+1, new_score[0]])
            # print(i, score, t)
    
    return routeplanning, scores
        

def simulated_annealing_multiplicative(mtsp, n, routeplanning=None, t_max=100, cooling_scheme="linear", cooling_parameter=1, step_type="insert"):
    """
    Cooling schemes: constant thermodynamic speed, exponential, logarithmic, and linear cooling schedules
    """
    # Select hillclimbing step
    steps = select_step_type(step_type)

    if cooling_scheme == "exponential":
        cooling_function = exponential_multiplicative_cooling
    elif cooling_scheme == "linear":
        cooling_function = linear_multiplicative_cooling

    scores = []
    if routeplanning == None:
        routeplanning = gen_random_planning(mtsp)
    
    score = routeplanning.calculate_score()[0]
    scores.append([0,score])
    t = t_max
    for i in range(n):
        new_planning, new_score = do_step(routeplanning.copy(), steps)

        if acceptance_function(t, score, new_score):
            routeplanning = new_planning
            score = new_score
            scores.append([i+1, new_score])
            # print(i, score)

        t = cooling_function(t, cooling_parameter)

def stochastic_hillclimber(mtsp, n, routeplanning=None, step_type=["insert"]):
    """ Main hill climbing algorithm loop for n evaluations, step_type is a list containing all mutation operators used """

    # Select hillclimbing step
    steps = select_step_type(step_type)

    scores = []
    if routeplanning == None:
        routeplanning = gen_random_planning(mtsp)

    score = routeplanning.calculate_score()
    scores.append([0,score[0]])
    for i in range(n):
        new_planning, new_score = do_step(routeplanning.copy(), steps)

        if new_score[0] <= score[0]:
            if new_score[0] < score[0]:
                scores.append([i+1, new_score[0]])
            
            routeplanning = new_planning
            score = new_score
                

    return routeplanning, scores

def counting_hillclimber(mtsp, n, routeplanning=None, step_type=["insert"]):
    """ 
    Specifically written for answering a research question, not the main algorithm. 
    Keeps track of which mutation operators are used if a mixture is used.
    """

    # Select hillclimbing step
    steps = select_step_type(step_type)

    scores = []
    if routeplanning == None:
        routeplanning = gen_random_planning(mtsp)

    score = routeplanning.calculate_score()
    scores.append([0,score[0]])
    for i in range(n):
        step = random.sample(step_type, 1)
        new_planning, new_score = do_step(routeplanning.copy(), select_step_type(step))

        if new_score[0] <= score[0]:
            if new_score[0] < score[0]:
                scores.append([i+1, new_score[0], step[0]])

            routeplanning = new_planning
            score = new_score
                

    return routeplanning, scores

def hillclimb_one_route(mtsp, n, route, step_type="random"):
    """ Hill climbing on TSP """

    # Select hillclimbing step
    step = select_step_type(step_type)

    score = route.calculate_score()
    for i in range(n):
        new_planning, new_score = step(route.copy())
        new_route = step(copy.copy(route))
        new_score = mtsp.route_length(new_route)
        if new_score <= score:
            route = new_route
            score = new_score

    return route

def hillclimb_routes_seperately(mtsp, n, routeplanning=None, step_type=["reverse"]):
    """ Local optimization on the routes seperately """

    if routeplanning == None:
        routeplanning = gen_random_fair_planning(mtsp)

    routes_list = routeplanning.split_to_lists()

    new_routes = []
    for route in routes_list:
        if len(route) > 1:
            new_route = stochastic_hillclimber(mtsp, n, Routeplanning(mtsp, [0] + route), step_type)[0].routes
            new_route.remove(0)
            new_routes.append(new_route)

    routeplanning.lists_to_permutation(new_routes)

    return routeplanning

    