import numpy as np
import csv
import os
import glob

from mtsp.scripts.helpers import gps_to_seconds, haversine

def read_points_file(points_file):
    points = []
    with open(points_file, "r", newline='') as f:
        reader = csv.reader(f, delimiter=",", quotechar="|")
        for row in reader:
            points.append([float(row[0]), float(row[1])])

    return points

def write_matrix(matrix, file_path, name):
    # Take the file out of the path
    out_path = os.path.join(*file_path.split(os.sep)[:-1], name)

    with open(out_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|')
        for row in matrix:
            writer.writerow(row)

def create_euclidean_matrix(file_path, style="km"):
    points = read_points_file(file_path)

    if style == "km":
        matrix = np.zeros(dtype=np.float64, shape=(len(points), len(points)))
    elif style == "seconds":
        matrix = np.zeros(dtype=np.int64, shape=(len(points), len(points)))

    for i in range(len(points)):
        for j in range(len(points)):
            if style == "km":
                matrix[i][j] = haversine(points[i], points[j])
            elif style == "seconds":
                matrix[i][j] = gps_to_seconds(points[i], points[j])
            else:
                raise ValueError()
                

    write_matrix(matrix, file_path, "euclidean.csv")

def create_euclidean_matrices(folder_path, style="km"):
    """
    Looks at all subfolders of folder_path that contain a points.csv file and create a euclidean.csv file in them, based on points
    """
    folder_path = os.path.join(folder_path, "**", "points.csv")
    paths = glob.glob(folder_path, recursive=True)
    for i, file_path in enumerate(paths):
        print(f"creating matrix {i}")
        create_euclidean_matrix(file_path, style)
