import numpy as np
import csv
import copy
import os
import collections

class MTSP:
    """ An instance of a MTSP problem specific to bicycle couriers. Contains all deliveries, resources, and constraints """

    def __init__(self, data_path, data_type="euclidean", n_couriers=None):
        """
        Initializes the MTSP instance from data files, given by a path a data type, as well as a courier count.
        """
        self.path = data_path
        self.deliveries, self.hub = self.get_deliveries_and_hub(os.path.join(data_path, "points.csv"))
        self.n_deliveries = len(self.deliveries)
        self.n_nodes = self.n_deliveries + 1

        self.data_type = data_type
        self.distance_matrix = self.get_distance_matrix(os.path.join(data_path, f"{data_type}.csv"))

        self.value_count = 0

        if n_couriers == None:
            self.couriers = self.n_deliveries//20 + 1
        else:
            self.couriers = n_couriers

        self.all_points = np.arange(-(self.couriers - 1), self.n_nodes).tolist()

    def get_node(self, node_id):
        return self.deliveries[node_id - 1]
    
    def get_location(self, node_id):
        return self.get_node(node_id).location
    
    def distance(self, n1, n2):
        """ Takes two nodes and returns the distance between them """
        return self.distance_matrix[max(0,n1)][max(0,n2)]

    def route_length(self, route):
        """ Gives the length of a sub tour, takes all nodes within the sub tour excluding the hub node """
        score = 0
        
        if len(route) == 0:
            return 0

        # Here the distance from the hub
        score += self.distance(0, route[0])

        for i, delivery in enumerate(route[:-1]):
            score += self.distance(route[i], route[i+1])

        score += self.distance(route[-1], 0)

        return score

    def planning_length(self, planning):
        """ Calculates the value of a planning. This is given by the length of the longest sub tour """
        self.objective_count += 1

        for i, location in enumerate(planning):
            if location <= 0:
                first_route = i
                break
        planning = planning[first_route:] + planning[:first_route]

        routes = []
        for location in planning:
            if location <= 0:
                routes.append([])
            else:
                routes[-1].append(location)
        
        scores = []
        for route in routes:
            scores.append(self.route_length(route))

        return max(scores), sum(scores)
    
    def route_center(self, route):
        """ Finds the euclidean center of a route given its nodes """
        if len(route) == 0:
            return self.get_location(0)
        totals = [0,0]
        for i, node in enumerate(route):
            loc = self.get_location(node)
            totals[0] += loc[0]
            totals[1] += loc[1]

        return totals[0] / len(route), totals[1] / len(route)

    def added_distance(self, first_delivery, added_delivery, second_delivery):
        """ Calculates how much longer it takes to go from node a->b->c instead of a->c, can be negative if triangle inequality not satisfied """ 
        
        added_distance = self.distance(first_delivery, added_delivery) \
            + self.distance(added_delivery, second_delivery) \
            - self.distance(first_delivery, second_delivery)

        return added_distance
    
    def added_distance_in_route(self, ix, route):
        """ Takes a route and node in that route, and returns how much longer that node makes that route """
        if ix == 0:
            first = 0
        else:
            first = route[ix-1]
        if ix == len(route)-1:
            last = 0
        else:
            last = route[ix+1]
        
        return self.added_distance(first, route[ix], last)
    
    def best_place_in_route(self, node, route):
        """ Takes a node and a route, and returns the place in the route where that node minimizes route length, as well as the extra length """
        if len(route) == 0:
            return 0, self.added_distance(0, node, 0)
        
        ad = self.added_distance(0, node, route[0])
        place = 0
        for i in range(len(route) - 1):
            ad_i = self.added_distance(route[i], node, route[i+1])
            if ad_i < ad:
                place = i + 1
                ad = ad_i

        ad_end = self.added_distance(route[-1], node, 0)
        if ad_end < ad:
            place = len(route) + 1
            ad = ad_end
        

        return place, ad
    
    def calculate_distance_matrix(self):
        """ Calculates a euclidian distance matrix """

        n_deliveries = len(self.deliveries)
        distance_matrix = np.zeros((n_deliveries+1, n_deliveries+1))

        for i in range(n_deliveries):
            # Calculate distance to and from hub
            x_to_hub = self.deliveries[i].location[0] - self.hub.location[0]
            y_to_hub = self.deliveries[i].location[1] - self.hub.location[1]
            distance_to_hub = np.sqrt(x_to_hub**2 + y_to_hub**2)
            distance_matrix[i+1][0] = distance_matrix[0][i+1] = distance_to_hub

            # Calculate distance between two deliveries
            for j in range(i + 1, n_deliveries):
                x_distance = self.deliveries[i].location[0] - self.deliveries[j].location[0]
                y_distance = self.deliveries[i].location[1] - self.deliveries[j].location[1]
                distance = np.sqrt(x_distance**2 + y_distance**2)
                distance_matrix[i+1][j+1] = distance_matrix[j+1][i+1] = distance

        return distance_matrix
    
    def check_validity(self, routes):
        """ Checks whether a route is a permutation of all nodes """
        return collections.Counter(routes) == collections.Counter(self.all_points)
    
    def get_deliveries_and_hub(self, points_path):
        """
        Takes a csv file containing gps locations, one per row in the form
        latitude, longitude
        The first row corresponds to the hub, the others to the deliveries
        """
        deliveries = []
        with open(points_path, "r") as f:
            reader = csv.reader(f)
            hub_location_str = next(reader)
            hub = Hub((float(hub_location_str[0]), float(hub_location_str[1])))
            # deliveries.append(hub)
            for i, row in enumerate(reader):
                deliveries.append(Delivery(i+1, (float(row[0]), float(row[1]))))

        return deliveries, hub
    
    def get_distance_matrix(self, distance_matrix_path):
        """
        Creates a distance matrix from a csv file containing the distance matrix
        """
        distance_matrix = np.zeros(shape=(self.n_deliveries+1, self.n_deliveries+1), dtype=np.float64)
        with open(distance_matrix_path, "r") as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                for j, value in enumerate(row):
                    distance_matrix[i][j] = float(value)

        return distance_matrix
        
class Routeplanning:
    """ A solution to the MTSP, contains a list that is the permutation over all nodes, as well as a function to calculate objective value"""
    def __init__(self, mtsp, routes=[]):
        """ Initializes the solution for a given mtsp """
        self.mtsp = mtsp

        self.routes = routes

        self.score = None
        self.total_score = None

    def calculate_score(self):
        """ Calculates the objective value of the solution, as well as the total length of all routes """
        self.score, self.total_score = self.mtsp.planning_length(self.routes)
        return self.score, self.total_score
    
    def split_to_lists(self):
        """
        Reads through the list of routes and splits it into a list of route-lists
        """
        routes_list = []

        # The first nodes seen are the last nodes of the last route, since the last route wraps around the end of the permutation
        final_route_tail = []

        seen_hub_node = 0
        for node in self.routes:
            if node <= 0:
                routes_list.append([])
                seen_hub_node = 1
            else:
                # Add nodes to current route if the hub has been seen, to the end of the last route if not
                if seen_hub_node:
                    routes_list[-1].append(node)
                else:
                    final_route_tail.append(node)

        # Concatenate the beginning and end of the final route
        routes_list[-1] = routes_list[-1] + final_route_tail

        return routes_list
    
    def lists_to_permutation(self, lists, update_score=False):
        """ Takes a list of routes, and adds them together to form a single permutation """
        if update_score:
            self.mtsp.objective_count += 1

        routes = []
        for i, list in enumerate(lists):
            routes = routes + [-i] + list

        self.routes = routes

        if self.mtsp.check_validity(routes) == False:
            print(routes)
            raise ValueError("routes are not correcto")

        return self.routes
    
    def copy(self, routes=None):
        if routes == None:
            routes = copy.copy(self.routes)
        return Routeplanning(self.mtsp, routes)

class Hub:
    """ The hub (depot) node"""
    def __init__(self, location):
        self.id = 0
        self.location = location


class Delivery:
    """ A delivery node"""
    def __init__(self, _id, location, postal_code=None, address=None):
        self.id = _id
        self.location = location

        self.postal_code = postal_code
        self.address = address

    def __repr__(self):
        return f"Delivery {self.id}"

