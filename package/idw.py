import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

def idw(readings, boundary, num_points_lon=100, power=2, epsilon=1e-10):
    """
    Function to perform Inverse Distance Weighted (IDW) interpolation using KD-Tree for improved efficiency.

    Parameters:
    readings (pd.DataFrame): DataFrame containing the readings with columns 'longitude', 'latitude', and 'value'.
    boundary (pd.DataFrame): DataFrame containing the grid boundaries with columns 'lon' and 'lat'.
    num_points_lon (int, optional): Number of points in the longitude direction of the grid. Default is 100.
    power (int, optional): Exponent used in the IDW formula to calculate weights. Default is 2.
    epsilon (float, optional): Small value to avoid division by zero in distances. Default is 1e-10.

    Returns:
    xi, yi, zi: Grid coordinates and interpolated values.
    """
    # Extract coordinates and values from readings
    lon = np.array(readings['longitude'])
    lat = np.array(readings['latitude'])
    value = np.array(readings['value'])

    # Determine grid extent from boundaries
    lon_min = boundary['lon'].min()
    lon_max = boundary['lon'].max()
    lat_min = boundary['lat'].min()
    lat_max = boundary['lat'].max()

    # Calculate the number of points in the latitude direction
    num_points_lat = int(num_points_lon * (lat_max - lat_min) / (lon_max - lon_min))

    # Create the grid
    x_grid = np.linspace(lon_min, lon_max, num_points_lon)
    y_grid = np.linspace(lat_min, lat_max, num_points_lat)
    xi, yi = np.meshgrid(x_grid, y_grid)

    # Flatten the grid for vectorized calculations
    xi_flat = xi.ravel()
    yi_flat = yi.ravel()

    # Create KD-Tree for the reading coordinates
    tree = cKDTree(np.vstack((lon, lat)).T)

    # Calculate distances from grid points to data points using KD-Tree
    distances, indices = tree.query(np.vstack((xi_flat, yi_flat)).T, k=len(lon))

    # Apply IDW formula to calculate weights
    distances = np.maximum(distances, epsilon)  # Avoid division by zero
    weights = 1.0 / distances**power
    weights /= weights.sum(axis=1)[:, None]

    # Calculate interpolated values
    zi_flat = np.sum(weights * value[indices], axis=1)

    # Reshape the results back to the original grid shape
    zi = zi_flat.reshape(xi.shape)

    return xi, yi, zi


def assign_sensor_ids(readings):
    """
    Function to assign sensor_id from 1 to 18 based on their actual values.
    The sensor with the highest value will get 18, and the one with the lowest value will get 1.

    Parameters:
    readings (pd.DataFrame): DataFrame containing the original readings with 'value' column.

    Returns:
    readings (pd.DataFrame): DataFrame with an additional 'sensor_id' column.
    """
    readings = readings.copy()
    # Rank the sensors based on their actual values (descending order)
    readings['sensor_id'] = (readings['value'].rank(method='first', ascending=False)
                             + (readings.index / 1000000)).rank(method='max').astype(int)

    return readings


def plot_idw(xi, yi, zi, readings, title='IDW Interpolation'):
    """
    Function to visualize the results of IDW interpolation, with sensor labels and a legend.

    Parameters:
    xi, yi, zi: Grid coordinates and interpolated values.
    readings (pd.DataFrame): DataFrame containing the original readings.
    title (str, optional): Title of the plot. Default is 'IDW Interpolation'.
    """
    plt.imshow(zi, extent=(xi.min(), xi.max(), yi.min(), yi.max()), origin='lower')
    plt.colorbar()

    # Plot the readings as scatter points
    sc = plt.scatter(readings['longitude'], readings['latitude'], facecolors='none', edgecolors='k', label='Location')

    # Add sensor_id labels to the scatter points
    for i, row in readings.iterrows():
        plt.text(row['longitude'], row['latitude'] + 0.0004,  # Adjust the vertical offset as needed
                 str(row['sensor_id']),
                 color='white', fontsize=8, ha='center', va='bottom', fontweight='bold')

    #plt.scatter([], [], marker='$1$', color='k', label='Sensor ID')
    #plt.legend(loc='upper left')

    legend_circle = plt.Line2D([0], [0], marker='o', color='none', markeredgecolor='black', markersize=6,
                               label='Known Point', markerfacecolor='none')

    # Create a red circle for masked points (only outline)
    masked_circle = plt.Line2D([0], [0], marker='o', color='none', markeredgecolor='red', markersize=6,
                               label='Masked Point', markerfacecolor='none')

    plt.legend(handles=[legend_circle, masked_circle], loc='upper left', fontsize='small')

    # Remove ticks but keep labels
    plt.xticks([])
    plt.yticks([])

    #plt.xlabel('Longitude')
    #plt.ylabel('Latitude')
    plt.title(title)
    plt.savefig('mappa.png')
    plt.show()


def calculate_rmse(test_data, interpolated_values, indices):
    return np.sqrt(np.sum((np.array(test_data['value']) - interpolated_values[indices]) ** 2) / test_data.shape[0])


def holdout_idw(readings, boundary, num_points_lon, power, holdout_frac, random_seed=0):
    np.random.seed(random_seed)
    rmse_per_timestamp = {}

    for timestamp, group in readings.groupby('timestamp'):
        if len(group) < 2:
            continue

        training_data = group.sample(frac=holdout_frac, random_state=random_seed)
        xi, yi, zi = idw(training_data, boundary, num_points_lon, power)

        grid_points = np.column_stack((xi.ravel(), yi.ravel()))
        interpolated_values = zi.ravel()

        test_data = group.drop(training_data.index)
        test_data.reset_index(drop=True, inplace=True)
        test_points = np.array(test_data[['longitude', 'latitude']])

        tree = cKDTree(grid_points)
        distances, indices = tree.query(test_points, k=1)

        rmse_per_timestamp[timestamp] = calculate_rmse(test_data, interpolated_values, indices)

    return rmse_per_timestamp


def kfoldcv_idw(readings, boundary, num_points_lon, power, k, random_seed=0):
    np.random.seed(random_seed)
    rmse_per_timestamp = {}

    for timestamp, group in readings.groupby('timestamp'):
        if len(group) < k:
            print(f"Timestamp {timestamp} ha meno di {k} punti, quindi verrà saltato.")
            continue

        group = group.reset_index(drop=True)
        indices = np.random.permutation(group.index)
        fold_indices = np.array_split(indices, k)

        rmse_list = []
        for fold_index in fold_indices:
            test_data = group.iloc[fold_index]
            training_data = group.drop(test_data.index)
            xi, yi, zi = idw(training_data, boundary, num_points_lon, power)

            grid_points = np.column_stack((xi.ravel(), yi.ravel()))
            interpolated_values = zi.ravel()

            tree = cKDTree(grid_points)
            distances, indices = tree.query(test_data[['longitude', 'latitude']], k=1)

            rmse_list.append(calculate_rmse(test_data, interpolated_values, indices))

        rmse_per_timestamp[timestamp] = np.mean(rmse_list)

    return rmse_per_timestamp