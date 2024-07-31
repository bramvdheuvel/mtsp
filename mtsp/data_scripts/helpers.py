import numpy as np

def haversine(loc1, loc2):
    """
    Takes two gps locations in the form of (latitude, longitude), and returns the distance between both locations in km
    """
    # Approximate radius of earth in km
    R = 6373.0

    lat1 = np.radians(loc1[0])
    lon1 = np.radians(loc1[1])
    lat2 = np.radians(loc2[0])
    lon2 = np.radians(loc2[1])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c

def distance_to_seconds(distance, speed=15):
    """
    Takes a distance in km, a speed in km/hour, and returns a time in seconds (int)
    """
    return int(3600 * distance / speed)

def gps_to_seconds(loc1, loc2, speed=15):
    return distance_to_seconds(haversine(loc1, loc2), speed)