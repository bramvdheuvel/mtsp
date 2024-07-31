import os
import numpy as np
import csv
import matplotlib.pyplot as plt

def points_to_data_to_points(folder):
    """ Changes the name of all points.csv files to data.csv, and creates new points files with only gps coordinates
    
    """
    for path, subdirs, files in os.walk(folder):
        for name in files:
            if name == "points.csv":
                points_file = os.path.join(path, name)
                data_file = os.path.join(path, "data.csv")
                os.rename(points_file, data_file)

    for path, subdirs, files in os.walk(folder):
        for name in files:
            if name == "data.csv":
                data_file = os.path.join(path, name)
                points_file = os.path.join(path, "points.csv")

                lat = []
                lon = []
                with open(data_file, 'r', newline='') as csvfile:
                    reader = csv.reader(csvfile, delimiter=';', quotechar='|')
                    next(reader)
                    for row in reader:
                        lat.append(float(row[-2]))
                        lon.append(float(row[-1]))

                with open(points_file, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',', quotechar='|')
                    for i in range(len(lat)):
                        writer.writerow([lat[i], lon[i]])

def swap_latitude_longitude(path):
    latitudes = []
    longitudes = []
    with open(path, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            latitudes.append(float(row[1]))
            longitudes.append(float(row[0]))

    with open(path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|')
        for i in range(len(latitudes)):
            writer.writerow([latitudes[i], longitudes[i]])

def get_dataset_boundaries(city):
    """ Returns a square area defined by gps coordinates for a number of dutch cities
    
    Params:
    city (string): A dutch city, possibilities are Almere, Amersfoort, Amsterdam, Groningen, Leeuwarden, Rotterdam, Zwolle
    Returns:
    list [[lat_min, lat_max], [lon_min, lon_max]]

    
    """
    try:
        area_file = os.path.join("datasets", "areas", f"{city}.csv")
    except:
        print("The city provided is not provided like it should be provided")

    points = []
    with open(area_file, "r", newline='') as f:
        reader = csv.reader(f, delimiter=",", quotechar="|")
        
        for row in reader:
            points.append([float(row[0]), float(row[1])])
        
    lines = []
    for i, point in enumerate(points):
        if i == 0:
            continue
        # Represent line as [[x0, y0], [x1, y1]]
        line = [[point[0], point[1]], [points[i-1][0], points[i-1][1]]]
        if line[0] != [0,0] and line[1] != [0,0]:
            lines.append(line)

    corners = []
    for corner in points:
        if corner != [0,0]:
            corners.append(corner)

    return lines, corners

def ccw(A, B, C):
    """https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/"""
    return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])

def intersect(l0, l1):
    """https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/"""
    return ccw(l0[0], l1[0], l1[1]) != ccw(l0[1],l1[0],l1[1]) and ccw(l0[0],l0[1],l1[0]) != ccw(l0[0],l0[1],l1[1])

def raycast(point, lines, max_longitude):
    """ Tests whether a point is inside a set of polygons by using raycasting 
    https://paulbourke.net/geometry/polygonmesh/
    https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/

    Params:
    point: a tuple (latitude: float, longitude: float)
    lines: a list of lines [[x0, y0], [x1, y1]]

    Returns:
    boolean; 1 if inside polygons, 0 if outside
    """
    # Draw line between point and max_longitude
    point_line = [[point[0], point[1]], [point[0], max_longitude]]

    # Count how many lines are crossed
    count = 0
    for line in lines:
        count += intersect(line, point_line)

    # Return lines crossed % 2
    return count % 2

def get_dataset_points(city, n):
    # Get dataset boundaries
    # print("Getting boundaries")
    lines, corners = get_dataset_boundaries(city)

    # Find sample zone
    # print("Creating sample area")
    corners = list(zip(*corners))
    lat_min = min(corners[0])
    lat_max = max(corners[0])
    lon_min = min(corners[1])
    lon_max = max(corners[1])
    lon_cast = lon_max + 1

    # Sample points and reject by raycasting
    # print("Sampling points")
    points = []
    rejected = []
    x = []
    y = []
    xr = []
    yr = []
    while len(points) < n:
        lat = np.random.uniform(lat_min, lat_max)
        lon = np.random.uniform(lon_min, lon_max)
        point = (lat, lon)

        if raycast(point, lines, lon_cast) == 1:
            points.append(point)

    return points

def make_dataset(city, n, out_path):
    points = get_dataset_points(city, n)


    with open(out_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|')
        for point in points:
            writer.writerow(point)

def make_datasets_for_city(city, point_range, n_per_point, n_first_dataset=0):
    for n_points in point_range:
        for i in range(n_first_dataset, n_per_point + n_first_dataset):
            out_folder = os.path.join("datasets", "synthetic", city, f"n{n_points}", str(i).zfill(4))
            os.makedirs(out_folder, exist_ok=True)
            out_path = os.path.join(out_folder, "points.csv")
            make_dataset(city, n_points, out_path)

def make_datasets_thesis():
    for i in range(1000):
        print(f"sampling points {i}")
        out_folder = os.path.join("datasets", "amsterdam", str(i).zfill(4))
        os.makedirs(out_folder, exist_ok=True)
        out_path = os.path.join(out_folder, "points.csv")
        make_dataset("amsterdam", 201, out_path)

def make_datasets_test():
    for i in range(100):
        print(f"sampling points {i}")
        out_folder = os.path.join("datasets", "test", str(i).zfill(4))
        os.makedirs(out_folder, exist_ok=True)
        out_path = os.path.join(out_folder, "points.csv")
        make_dataset("amsterdam", 16, out_path)
