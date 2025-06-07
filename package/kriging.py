import numpy as np
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging
from scipy.spatial import cKDTree


def kriging(readings, boundary, num_points_lon=100, nlags=100):
    lon = np.array(readings['longitude'])
    lat = np.array(readings['latitude'])
    value = np.array(readings['value'])

    # The extent of the grid
    lon_min = min(np.array(boundary['lon']))
    lon_max = max(np.array(boundary['lon']))
    lat_min = min(np.array(boundary['lat']))
    lat_max = max(np.array(boundary['lat']))

    # Determine the number of points in y axis of the grid
    num_points_lat = int(num_points_lon * (lat_max - lat_min) / (lon_max - lon_min))

    # Create the variogram
    OK = OrdinaryKriging(
        lon,
        lat,
        value,
        variogram_model='exponential',
        nlags=nlags,
        weight=False,
        verbose=False,
        enable_plotting=False,
        coordinates_type='geographic',
        exact_values=True
    )

    if OK.variogram_model_parameters[1] < np.min(OK.get_variogram_points()[0]):
        OK = OrdinaryKriging(
            lon,
            lat,
            value,
            variogram_model='exponential',
            nlags=10 * nlags,
            weight=False,
            verbose=False,
            enable_plotting=False,
            coordinates_type='geographic',
            exact_values=True
        )

    if OK.variogram_model_parameters[1] < np.min(OK.get_variogram_points()[0]):
        OK = OrdinaryKriging(
            lon,
            lat,
            value,
            variogram_model='exponential',
            nlags=100 * nlags,
            weight=False,
            verbose=False,
            enable_plotting=False,
            coordinates_type='geographic',
            exact_values=True
        )

    if OK.variogram_model_parameters[1] < np.min(OK.get_variogram_points()[0]):
        OK = OrdinaryKriging(
            lon,
            lat,
            value,
            variogram_model='spherical',
            nlags=nlags,
            weight=False,
            verbose=False,
            enable_plotting=False,
            coordinates_type='geographic',
            exact_values=True
        )

    if OK.variogram_model_parameters[1] < np.min(OK.get_variogram_points()[0]):
        OK = OrdinaryKriging(
            lon,
            lat,
            value,
            variogram_model='spherical',
            nlags=10 * nlags,
            weight=False,
            verbose=False,
            enable_plotting=False,
            coordinates_type='geographic',
            exact_values=True
        )

    if OK.variogram_model_parameters[1] < np.min(OK.get_variogram_points()[0]):
        OK = OrdinaryKriging(
            lon,
            lat,
            value,
            variogram_model='spherical',
            nlags=100 * nlags,
            weight=False,
            verbose=False,
            enable_plotting=False,
            coordinates_type='geographic',
            exact_values=True
        )

    if OK.variogram_model_parameters[1] < np.min(OK.get_variogram_points()[0]):
        OK = OrdinaryKriging(
            lon,
            lat,
            value,
            variogram_model='gaussian',
            nlags=nlags,
            weight=False,
            verbose=False,
            enable_plotting=False,
            coordinates_type='geographic',
            exact_values=True
        )

    if OK.variogram_model_parameters[1] < np.min(OK.get_variogram_points()[0]):
        OK = OrdinaryKriging(
            lon,
            lat,
            value,
            variogram_model='gaussian',
            nlags=10 * nlags,
            weight=False,
            verbose=False,
            enable_plotting=False,
            coordinates_type='geographic',
            exact_values=True
        )

    if OK.variogram_model_parameters[1] < np.min(OK.get_variogram_points()[0]):
        OK = OrdinaryKriging(
            lon,
            lat,
            value,
            variogram_model='gaussian',
            nlags=100 * nlags,
            weight=False,
            verbose=False,
            enable_plotting=False,
            coordinates_type='geographic',
            exact_values=True
        )

    # Create the grid
    x_grid = np.linspace(lon_min, lon_max, num_points_lon)
    y_grid = np.linspace(lat_min, lat_max, num_points_lat)
    xi, yi = np.meshgrid(x_grid, y_grid)
    zi, ss = OK.execute("grid", x_grid, y_grid)
    '''
    # Plot the results
    plt.imshow(zi, extent=(lon_min, lon_max, lat_min, lat_max), origin='lower', cmap='viridis')
    plt.colorbar(label='Interpolated Values')
    plt.scatter(lon, lat, c=value, edgecolors='k', cmap='viridis')
    plt.title('Kriging Interpolation '+'at '+readings.iloc[0][3])
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.show()
    '''
    return xi, yi, zi


def plot_ok(xi, yi, zi, readings, title='Ordinary Kriging Interpolation'):
    """
    Function to visualize the results of IDW interpolation.

    Parameters:
    xi, yi, zi: Grid coordinates and interpolated values.
    readings (pd.DataFrame): DataFrame containing the original readings.
    title (str, optional): Title of the plot. Default is 'IDW Interpolation'.
    """
    plt.imshow(zi, extent=(xi.min(), xi.max(), yi.min(), yi.max()), origin='lower', cmap='viridis')
    plt.colorbar(label='Interpolated Values')
    plt.scatter(readings['longitude'], readings['latitude'], c=readings['value'], edgecolors='k', cmap='viridis')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(title)
    plt.show()


def calculate_rmse(test_data, interpolated_values, indices):
    return np.sqrt(np.sum((np.array(test_data['value']) - interpolated_values[indices]) ** 2) / test_data.shape[0])


def holdout_kriging(readings, boundary, num_points_lon, nlags, holdout_frac, random_seed=0):
    np.random.seed(random_seed)
    rmse_per_timestamp = {}

    for timestamp, group in readings.groupby('timestamp'):
        if len(group) < 2:
            continue

        training_data = group.sample(frac=holdout_frac, random_state=random_seed)
        xi, yi, zi = kriging(training_data, boundary, num_points_lon, nlags)

        grid_points = np.column_stack((xi.ravel(), yi.ravel()))
        interpolated_values = zi.ravel()

        test_data = group.drop(training_data.index)
        test_data.reset_index(drop=True, inplace=True)
        test_points = np.array(test_data[['longitude', 'latitude']])

        tree = cKDTree(grid_points)
        distances, indices = tree.query(test_points, k=1)

        rmse_per_timestamp[timestamp] = calculate_rmse(test_data, interpolated_values, indices)

    return rmse_per_timestamp


def kfoldcv_kriging(readings, boundary, num_points_lon, nlags, k, random_seed=0):
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
            xi, yi, zi = kriging(training_data, boundary, num_points_lon, nlags)

            grid_points = np.column_stack((xi.ravel(), yi.ravel()))
            interpolated_values = zi.ravel()

            tree = cKDTree(grid_points)
            distances, indices = tree.query(test_data[['longitude', 'latitude']], k=1)

            rmse_list.append(calculate_rmse(test_data, interpolated_values, indices))

        rmse_per_timestamp[timestamp] = np.mean(rmse_list)

    return rmse_per_timestamp


def kfoldcv_per_timestamp_kriging(readings, boundary, num_points_lon, nlags, k, random_seed=0):
    np.random.seed(random_seed)

    rmse_per_timestamp = {}

    for timestamp, group in readings.groupby('timestamp'):
        if len(group) < k:
            print(f"Timestamp {timestamp} ha meno di {k} punti, quindi verrà saltato.")
            continue

        group = group.reset_index(drop=True)

        indices = np.random.permutation(group.index)
        fold_size = len(indices) // k
        fold_indices = [indices[i * fold_size: (i + 1) * fold_size] for i in range(k)]

        if len(indices) % k != 0:
            fold_indices[-1] = np.concatenate((fold_indices[-1], indices[k * fold_size:]))

        folds = [group.iloc[fold_index] for fold_index in fold_indices]

        rmse_list = []
        for i in range(k):
            test_data = folds[i]
            test_points = np.array(test_data[['longitude', 'latitude']])

            training_data = group.drop(test_data.index)

            xi, yi, zi = kriging(training_data, boundary, num_points_lon, nlags)

            grid_points = np.column_stack((xi.ravel(), yi.ravel()))
            interpolated_values = zi.ravel()

            tree = cKDTree(grid_points)
            distances, indices = tree.query(test_points, k=1)

            fold_rmse = np.sqrt(
                np.sum((np.array(test_data['value']) - interpolated_values[indices]) ** 2) / test_data.shape[0])
            rmse_list.append(fold_rmse)

        rmse_per_timestamp[timestamp] = np.mean(rmse_list)

    return rmse_per_timestamp