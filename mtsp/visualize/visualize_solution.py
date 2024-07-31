import gmplot
import csv
import matplotlib.pyplot as plt

def create_points_list(points_file):
    pass

def visualize_points_simple(points_file):
    lat = []
    lon = []
    with open(points_file, "r", newline='') as f:
        reader = csv.reader(f, delimiter=",", quotechar="|")
        for row in reader:
            lat.append(float(row[0]))
            lon.append(float(row[1]))

    plt.plot(lon, lat, 'ro')
    plt.show()

def visualize_solution_simple(routeplanning):
    # Plot points
    plt.plot(routeplanning.mtsp.hub.location[1], routeplanning.mtsp.hub.location[0], 'bo')
    lat = []
    lon = []
    for delivery in routeplanning.mtsp.deliveries:
        lat.append(delivery.location[0])
        lon.append(delivery.location[1])
    plt.plot(lon, lat, 'ro')

    route_lat = [routeplanning.mtsp.hub.location[0]]
    route_lon = [routeplanning.mtsp.hub.location[1]]
    print(len(routeplanning.routes))
    for i, location in enumerate(routeplanning.routes):
        if location <= 0:
            first_route = i
            break
    planning = routeplanning.routes[first_route:] + routeplanning.routes[:first_route]
    route = []
    for delivery in planning[1:]:
        # if delivery == 0:
        #     print(routeplanning.mtsp.deliveries[delivery].location)
        #     print(routeplanning.mtsp.deliveries[delivery].id)
        if delivery > 0:
            location = routeplanning.mtsp.deliveries[delivery-1].location
            route_lat.append(location[0])
            route_lon.append(location[1])
            route.append(delivery)
        else:
            print("NNNNNNN", routeplanning.mtsp.route_length(route))
            route_lat.append(routeplanning.mtsp.hub.location[0])
            route_lon.append(routeplanning.mtsp.hub.location[1])
            plt.plot(route_lon, route_lat, label=routeplanning.mtsp.route_length(route))
            route = []
            route_lat = [routeplanning.mtsp.hub.location[0]]
            route_lon = [routeplanning.mtsp.hub.location[1]]
    route_lat.append(routeplanning.mtsp.hub.location[0])
    route_lon.append(routeplanning.mtsp.hub.location[1])
    plt.plot(route_lon, route_lat, label=routeplanning.mtsp.route_length(route))
    
    plt.legend()
    plt.show()

def visualize_points_gmplot(points_file):
    apikey = '' # (your API key here)
    gmap = gmplot.GoogleMapPlotter(37.766956, -122.438481, 13, apikey=apikey)

    gmap.marker(37.793575, -122.464334, label='H', info_window="<a href='https://www.presidio.gov/'>The Presidio</a>")
    gmap.marker(37.768442, -122.441472, color='green', title='Buena Vista Park')
    gmap.marker(37.783333, -122.439494, precision=2, color='#FFD700')

    gmap.draw('map.html')