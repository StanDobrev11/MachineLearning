import numpy as np



# Mercator latitudes
def mercator_latitude(lat):
    return np.log(np.tan(np.pi / 4 + lat / 2))


def mercator_conversion(lat1, lon1, lat2, lon2):
    # Convert degrees to radians
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    lon1 = np.radians(lon1)
    lon2 = np.radians(lon2)

    delta_phi = mercator_latitude(lat2) - mercator_latitude(lat1)

    # Difference in longitudes
    delta_lambda = lon2 - lon1

    return delta_phi, delta_lambda


def rumbline_distance(lat1, lon1, lat2, lon2):
    delta_phi, delta_lambda = mercator_conversion(lat1, lon1, lat2, lon2)

    # Calculate distance using the Mercator Sailing formula
    return np.sqrt((delta_lambda * np.cos(np.radians(lat1))) ** 2 + delta_phi ** 2) * 3440.065


def bearing_to_waypoint(lat1, lon1, lat2, lon2):
    delta_phi, delta_lambda = mercator_conversion(lat1, lon1, lat2, lon2)

    # Calculate the bearing using atan2
    bearing_rad = np.arctan2(delta_lambda, delta_phi)

    # Convert from radians to degrees
    bearing_deg = np.degrees(bearing_rad)

    # Normalize to 0-360 degrees
    bearing_deg = (bearing_deg + 360) % 360

    return bearing_deg


def mercator_sailing_future_position(lat, lon, speed, bearing, time_interval):
    """
    Calculate future position given current lat, lon, speed (knots), bearing, and time interval (hours).

    Parameters:
    - lat (float): Current latitude in degrees
    - lon (float): Current longitude in degrees
    - speed (float): Speed in knots (nautical miles per hour)
    - bearing (float): Bearing in degrees (from north)
    - time_interval (float): Time interval in hours

    Returns:
    - new_lat (float): New latitude in degrees
    - new_lon (float): New longitude in degrees
    """
    # Convert degrees to radians
    lat = np.radians(lat)
    lon = np.radians(lon)
    bearing = np.radians(bearing)

    # Earth radius in nautical miles
    R = 3440.065

    # Distance traveled in nautical miles
    distance = speed * time_interval

    # Delta latitude (in radians)
    delta_lat = distance * np.cos(bearing) / R

    # Update latitude
    new_lat = lat + delta_lat

    # Delta longitude (in radians)
    if np.cos(new_lat) != 0:
        delta_lon = (distance * np.sin(bearing)) / (R * np.cos(new_lat))
    else:
        delta_lon = 0

    # Update longitude
    new_lon = lon + delta_lon

    # Convert radians back to degrees
    new_lat = np.degrees(new_lat)
    new_lon = np.degrees(new_lon)

    return np.array([new_lat, new_lon])



if __name__ == '__main__':
    lat1 = 0.0
    lon1 = 0.05
    lat2 = 0.2
    lon2 = 0.2
    print(rumbline_distance(lat1, lon1, lat2, lon2))
    print(bearing_to_waypoint(lat1, lon1, lat2, lon2))
    # print(mercator_sailing_future_position(lat1, lon1, 14, 0, 43.1))