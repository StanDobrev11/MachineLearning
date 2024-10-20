import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from navi import plane_sailing_course_speed, plane_sailing_next_position


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
    df = df.assign(x=np.cos(lat_rad) * np.cos(lon_rad))
    df = df.assign(y=np.cos(lat_rad) * np.sin(lon_rad))
    df = df.assign(z=np.sin(lat_rad))

    return df


def cartesian_to_geo(x, y, z):
    """
    Transforms Cartesian coordinates (x, y, z) back to latitude and longitude.

    Parameters:
        df (pandas.DataFrame): DataFrame with 'x', 'y', 'z' columns.

    Returns:
        pandas.DataFrame: DataFrame with added 'lat' and 'lon' columns.
    """
    # Calculate latitude (in radians)
    lat = np.round(np.degrees(np.arcsin(z)), 1)

    # Calculate longitude (in radians), using arctan2 to handle all quadrants
    lon = np.round(np.degrees(np.arctan2(y, x)), 1)

    return lat, lon


def calculate_velocity_direction(df):
    """
    Calculates velocity (kn) and direction (bearing in degrees) between consecutive points.
    """
    df = df.copy()
    
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

        # Calculate time difference in hours
        time_diff = (df.index[i] - df.index[i - 1]).total_seconds() / 3600

        # check time difference more than 6 hrs
        if time_diff > 6 or time_diff == 0:
            velocity[i] = 0
            direction[i] = 0
            continue

        # Calculate direction and velocity
        direction[i], velocity[i] = plane_sailing_course_speed(prev_point, curr_point, time_interval=time_diff)

    df['velocity_kn'] = velocity
    df['direction_deg'] = direction

    return df


def convert_direction_to_sin_cosin(df):
    direction_rad = np.deg2rad(df['direction_deg'])

    # Create sine and cosine components
    df = df.assign(direction_sin=np.sin(direction_rad))
    df = df.assign(direction_cos=np.cos(direction_rad))

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

    return int(angle_deg)


def generate_training_dataframe(df):
    def shift_group(df_group):
        """
        Shift features to create target variables for the next observation within each group (TD).
        """
        df_group['next_x'] = df_group['x'].shift(-1)
        df_group['next_y'] = df_group['y'].shift(-1)
        df_group['next_z'] = df_group['z'].shift(-1)
        df_group['next_max_wind_kn'] = df_group['max_wind_kn'].shift(-1)
        df_group['next_min_pressure_mBar'] = df_group['min_pressure_mBar'].shift(-1)
        df_group['next_velocity_kn'] = df_group['velocity_kn'].shift(-1)
        df_group['next_direction_sin'] = df_group['direction_sin'].shift(-1)
        df_group['next_direction_cos'] = df_group['direction_cos'].shift(-1)
        return df_group

    # Shift features within each group to create targets
    df = df.groupby('group').apply(shift_group, include_groups=False)

    # Reset index and drop unnecessary columns
    df = df.reset_index()
    df.index = pd.to_datetime(df['date'])
    df = df.drop(columns=['date'])

    # drop NaN containing lines
    df = df.dropna()

    return df


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
    8. Generates training data by lagging one observation.

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

    def shift_velocity_direction(df_group):
        """
        Shifts velocity and direction within each TD group (id), ensuring that the last observation has 0 velocity and 0 direction.
        """
        # Shift velocity and direction for each group
        df_group['velocity_kn'] = df_group['velocity_kn'].shift(-1)
        df_group['direction_deg'] = df_group['direction_deg'].shift(-1)

        # Replace NaN values in the last observation of each group with 0 (because TD dissipates)
        df_group[['velocity_kn', 'direction_deg']] = df_group[
            ['velocity_kn', 'direction_deg']].fillna(0)

        return df_group

    # Check if dataframe already has been transformed:
    if not all(col in df.columns for col in ['id', 'velocity_kn', 'direction_deg', 'enso']):
        # transform index column to datetime
        df.index = pd.to_datetime(df.index)

        # read enso data and add the feature to df
        enso_df = pd.read_csv('data/csv_ready/enso_years.csv')
        df.loc[:, 'enso'] = df.apply(import_enso_to_df, axis=1, enso_df=enso_df)

        # fix the NaN of the TDs name
        df.loc[:, 'name'] = df['name'].apply(lambda x: 'UNNAMED' if pd.isna(x) else x)

        df = df[['name', 'lat', 'lon', 'max_wind_kn', 'min_pressure_mBar', 'enso']]

        df = calculate_velocity_direction(df)

        # adding consecutive count of the TDs in order to be able to group and split the datasets
        new_td_starts = (df['velocity_kn'] == 0) & (df['direction_deg'] == 0)
        ids = new_td_starts.cumsum()
        df['group'] = ids

        df = df.groupby('group').apply(shift_velocity_direction)

        # Drop the 'group' column temporarily to avoid conflict when resetting the index
        df = df.drop(columns=['group'])

        # Reset index and drop unnecessary columns
        df = df.reset_index()
        df.index = pd.to_datetime(df['date'])
        df = df.drop(columns=['date'])

        # Add back the 'group' column after resetting the index
        df['group'] = ids

    df = convert_direction_to_sin_cosin(df)
    df = geo_to_cartesian(df)

    return df


def split_dataframe(df, splitter='gss', n_splits=1, test_size=0.2, random_state=97):
    # Define the features and target columns
    X = df[['x', 'y', 'z', 'max_wind_kn', 'min_pressure_mBar', 'velocity_kn',
            'direction_sin', 'direction_cos', 'enso']]
    y = df[['next_x', 'next_y', 'next_z', 'next_max_wind_kn', 'next_min_pressure_mBar',
            'next_velocity_kn', 'next_direction_sin', 'next_direction_cos']]

    # Use the specified splitter
    if splitter == 'gkf':
        splitter = GroupKFold(n_splits=n_splits)
    elif splitter == 'gss':
        splitter = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    else:
        raise ValueError("Invalid splitter type. Use 'gkf' for GroupKFold or 'gss' for GroupShuffleSplit.")

    groups = df['group']

    # Perform the split and yield train/test sets
    for train_idx, test_idx in splitter.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        yield X_train, X_test, y_train, y_test


def manage_prediction(df, model):
    predicted_values = model.predict(df)
    x, y, z, max_wind_kn, min_pressure_mBar, velocity_kn, direction_sin, direction_cos = predicted_values[0]

    lat, lon = cartesian_to_geo(x, y, z)
    min_pressure_mBar = int(min_pressure_mBar)
    max_wind_kn = np.round(max_wind_kn, 1)
    velocity_kn = np.round(velocity_kn, 1)
    direction_deg = get_direction_from_sin_cos(direction_sin, direction_cos)

    return lat, lon, max_wind_kn, min_pressure_mBar, velocity_kn, direction_deg
