import numpy as np
import pandas as pd

from navi import bearing_to_waypoint, rumbline_distance


def geo_to_cartesian(df):
    """
    Transforms latitude and longitude into Cartesian coordinates (x, y, z).

    Parameters:
        df (pandas.DataFrame): DataFrame with 'lat' and 'lon' columns.

    Returns:
        pandas.DataFrame: DataFrame with added 'x', 'y', 'z' columns.
    """
    # Convert lat/lon from degrees to radians
    lat_rad = np.radians(df['lat'])
    lon_rad = np.radians(df['lon'])

    # Compute x, y, z using spherical to Cartesian transformation
    df['x'] = np.cos(lat_rad) * np.cos(lon_rad)
    df['y'] = np.cos(lat_rad) * np.sin(lon_rad)
    df['z'] = np.sin(lat_rad)

    # df = df.drop(columns=['lat', 'lon'])

    return df


def cartesian_to_geo(df):
    """
    Transforms Cartesian coordinates (x, y, z) back to latitude and longitude.

    Parameters:
        df (pandas.DataFrame): DataFrame with 'x', 'y', 'z' columns.

    Returns:
        pandas.DataFrame: DataFrame with added 'lat' and 'lon' columns.
    """
    # Calculate latitude (in radians)
    df['lat'] = np.degrees(np.arcsin(df['z']))

    # Calculate longitude (in radians), using arctan2 to handle all quadrants
    df['lon'] = np.degrees(np.arctan2(df['y'], df['x']))

    df = df.drop(columns=['x', 'y', 'z'])

    return df


def calculate_velocity_direction(df):
    """
    Calculates velocity (kn) and direction (bearing in degrees) between consecutive points.
    """
    velocity = np.zeros(len(df))
    direction = np.zeros(len(df))

    for i in range(len(df)):
        # velocity and direction for the first record
        if i == 0:
            velocity[i] = 0
            direction[i] = 0
            continue

        # check previous name is different from current name
        previous_name = df.iloc[i - 1]['name']
        current_name = df.iloc[i]['name']

        if current_name != previous_name:
            velocity[i] = 0
            direction[i] = 0
            continue

            # Get the previous and current coordinates
        prev_point = (df.iloc[i - 1]['lat'], df.iloc[i - 1]['lon'])
        curr_point = (df.iloc[i]['lat'], df.iloc[i]['lon'])

        # Calculate distance in nautical miles
        distance = rumbline_distance(prev_point, curr_point)

        # Calculate time difference in hours
        time_diff = (df.index[i] - df.index[i - 1]).total_seconds() / 3600

        # check time difference more than 6 hrs
        if time_diff > 6:
            velocity[i] = 0
            direction[i] = 0
            continue

            # Calculate velocity in kn
        if distance == 0 or time_diff == 0:
            velocity[i] = 0
        else:
            velocity[i] = distance / time_diff

        # Calculate direction (bearing)
        direction[i] = bearing_to_waypoint(prev_point, curr_point)

    df.loc[:, 'velocity_kn'] = velocity
    df.loc[:, 'direction_deg'] = direction

    return df


def convert_direction_to_sin_cosin(df):
    direction_rad = np.deg2rad(df['direction_deg'])

    # Create sine and cosine components
    df['direction_sin'] = np.sin(direction_rad)
    df['direction_cos'] = np.cos(direction_rad)

    # Drop the original direction column if it's no longer needed
    # df = df.drop(columns=['direction_deg'])

    return df


def get_direction_from_sin_cos(sin_val, cos_val):
    """
    Reversed bearing in degrees from sin and cosin values.

    Usage:
    df['direction_deg_reversed'] = df.apply(lambda row: get_direction_from_sin_cos(row['direction_sin'], row['direction_cos']), axis=1)
    """
    # Get the angle in radians
    angle_rad = np.arctan2(sin_val, cos_val)

    # Convert radians to degrees
    angle_deg = np.degrees(angle_rad)

    # Make sure the angle is within the range 0° to 360°
    if angle_deg < 0:
        angle_deg += 360

    return angle_deg


def prepare_dataframe(df):
    """
    Prepares the input DataFrame for further analysis and machine learning modeling by performing several transformations:

    1. Transforms the index to a `datetime` format.
    2. Reads the ENSO (El Niño Southern Oscillation) phase data from a CSV file and adds an 'enso' feature to the DataFrame.
    3. Fills missing tropical depression names with 'UNNAMED' where applicable.
    4. Keeps only relevant columns: 'name', 'lat', 'lon', 'max_wind_kn', 'min_pressure_mBar', and 'enso'.
    5. Calculates velocity and direction based on latitude and longitude.
    6. Converts directional degrees into sine and cosine components for better model representation.
    7. Converts geographical coordinates (latitude, longitude) into 3D Cartesian coordinates (x, y, z).

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame containing tropical depression data with columns including latitude, longitude, wind speed, pressure, and name.

    Returns:
    --------
    pd.DataFrame
        A cleaned and transformed DataFrame with the following columns:
        - 'name': The name of the tropical depression.
        - 'lat': Latitude of the tropical depression.
        - 'lon': Longitude of the tropical depression.
        - 'max_wind_kn': Maximum wind speed (in knots).
        - 'min_pressure_mBar': Minimum central pressure (in millibars).
        - 'enso': The ENSO phase (1 for El Niño, -1 for La Niña, and 0 for Neutral).
        - 'velocity_kn': Velocity of the tropical depression (calculated from lat/lon).
        - 'direction_sin': Sine of the direction angle.
        - 'direction_cos': Cosine of the direction angle.
        - 'x', 'y', 'z': Cartesian coordinates of the tropical depression based on latitude and longitude.
    """

    def import_enso_to_df(row, enso_df):
        """
        Retrieves the ENSO (El Niño Southern Oscillation) phase for a given row based on its year.

        This function is designed to be applied row-wise to a DataFrame, and it assigns an ENSO phase (El Niño, La Niña, or Neutral)
        to each record based on the year of the tropical depression. ENSO phase values are drawn from a separate DataFrame containing
        the ENSO data for each year.

        Parameters:
        ----------
        row : pd.Series
            A row of the DataFrame being processed. The index of the row should contain datetime values from which the year can be extracted.

        enso_df : pd.DataFrame
            A DataFrame containing ENSO phase information, where each row corresponds to a specific year and includes an 'enso' column
            that holds the ENSO phase (-1 for La Niña, 0 for Neutral, 1 for El Niño).

        Returns:
        --------
        int
            The ENSO phase for the corresponding year of the input row. The function returns:
            - -1 for La Niña,
            - 0 for Neutral,
            - 1 for El Niño.
            If the year is 1949 (for specific handling), the function returns -1 (La Niña as a placeholder).

        Usage:
        ------
        df['enso'] = df.apply(import_enso_to_df, axis=1, enso_df=enso_df)

        Notes:
        ------
        - This function assumes that the DataFrame's index is of `datetime` type, allowing access to the year using `row.name.year`.
        """
        year = row.name.year
        if year == 1949:
            return -1
        return enso_df.loc[enso_df['year'] == year].enso.values[0]

    # transform index column to datetime
    df.index = pd.to_datetime(df.index)

    # read enso data and add the feature to df
    enso_df = pd.read_csv('data/csv_ready/enso_years.csv')
    df['enso'] = df.apply(import_enso_to_df, axis=1, enso_df=enso_df)

    # fix the NaN of the TDs name
    df['name'] = df['name'].apply(lambda x: 'UNNAMED' if pd.isna(x) else x)

    df = df[['name', 'lat', 'lon', 'max_wind_kn', 'min_pressure_mBar', 'enso']]

    df = calculate_velocity_direction(df)
    df = convert_direction_to_sin_cosin(df)
    df = geo_to_cartesian(df)

    return df
