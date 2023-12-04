import numpy as np
import xarray as xr
from skimage.draw import line
from scipy.interpolate import make_interp_spline
from scipy.ndimage import uniform_filter
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Function to draw a line on the image
def draw_line(image, angle, centerx, centery):
    center = (centerx, centery)
    angle = np.radians(angle)

    dx = np.cos(angle)
    dy = np.sin(angle)

    t_values = [
        (0 - center[0]) / dx,
        (image.shape[0] - center[0]) / dx,
        (0 - center[1]) / dy,
        (image.shape[1] - center[1]) / dy
    ]

    t_values.sort()
    t1, t2 = t_values[1:3]

    start_point = center + np.array([dx * t1, dy * t1])
    end_point = center + np.array([dx * t2, dy * t2])

    line_coords = line(int(start_point[0]), int(start_point[1]), int(end_point[0]), int(end_point[1]))
    line_coords = (np.clip(line_coords[0], 0, image.shape[0] - 1), np.clip(line_coords[1], 0, image.shape[1] - 1))

    heights = image[line_coords]

    return heights, line_coords

# Load your netCDF file
ds = xr.open_dataset('D:/Downloads/wassfast_output (2).nc')

# Define the area of interest
x_start, x_end = 50, 100
y_start, y_end = 130, 180

# Initialize lists to hold the mean heights and IMU data
mean_heights = []
imu_data = []

# Conversion factor from original units to meters
conversion_factor = 0.6 / 600

# Loop over the first 600 frames for training
for t in range(600):
    # Select the entire frame
    da = ds['Z'].isel(count=t)
    height_map = da.values
    height_map = np.nan_to_num(height_map, nan=0)

    heights, line_coords = draw_line(height_map, -177, 128, 128)
    mean_heights_frame = []
    for k in range(len(line_coords[0])):
        orthogonal_heights, _ = draw_line(height_map, -177+90, line_coords[0][k], line_coords[1][k])
        mean_heights_frame.append(np.nanmean(orthogonal_heights))

    x = np.array(line_coords[0])
    y = np.array(mean_heights_frame)
    print(t)
    unique_x, indices = np.unique(x, return_index=True)
    unique_y = y[indices]

    num_points = 1000
    xnew = np.linspace(unique_x.min(), unique_x.max(), num_points)

    spl = make_interp_spline(unique_x, unique_y, k=3)
    y_smooth = spl(xnew)
    mean_heights_frame = np.array(y_smooth)

    # Smooth the curve using a uniform filter
    smoothed_mean_heights_frame = uniform_filter(mean_heights_frame, size=5)
    # Append to our list
    mean_heights.append(smoothed_mean_heights_frame[:len(smoothed_mean_heights_frame)//2])

    # Calculate the mean height of the specified area and convert to meters
    area = height_map[x_start:x_end, y_start:y_end]
    for i in range(area.shape[0]):
        for j in range(area.shape[1]):
            if np.isnan(area[i][j]):
                area[i][j] = 0 
    mean_height = np.mean(area) * conversion_factor

    # Append to IMU data list
    imu_data.append(mean_height)
smoothed_imu = uniform_filter(imu_data, size=5)

# Prepare the input data for the Random Forest model
input_data_rf = np.array(mean_heights)
output_data_rf = np.array(smoothed_imu)  # This is your simulated IMU position

# Check if the input_data_rf is not empty before training the Random Forest model
if input_data_rf.shape[0] > 0:
    # Train the Random Forest model
    model_rf = RandomForestRegressor(n_estimators=200, random_state=0)
    model_rf.fit(input_data_rf, output_data_rf)

    # Initialize lists to hold the predicted IMU positions
    predicted_imu_rf = []

    # Loop over the frames for prediction
    for t in range(600, ds.dims['count']):
        # Select the entire frame
        da = ds['Z'].isel(count=t)
        height_map = da.values
        height_map = np.nan_to_num(height_map, nan=0)
        
        heights, line_coords = draw_line(height_map, -177, 128, 128)
        mean_heights_frame = []
        for k in range(len(line_coords[0])):
            orthogonal_heights, _ = draw_line(height_map, -177+90, line_coords[0][k], line_coords[1][k])
            mean_heights_frame.append(np.nanmean(orthogonal_heights))

        x = np.array(line_coords[0])
        y = np.array(mean_heights_frame)

        unique_x, indices = np.unique(x, return_index=True)
        unique_y = y[indices]

        num_points = 1000
        xnew = np.linspace(unique_x.min(), unique_x.max(), num_points)

        spl = make_interp_spline(unique_x, unique_y, k=3)
        y_smooth = spl(xnew)
        mean_heights_frame = np.array(y_smooth)

        # Smooth the curve using a uniform filter
        smoothed_mean_heights_frame = uniform_filter(mean_heights_frame, size=5)
        predicted_imu_rf.append(model_rf.predict(smoothed_mean_heights_frame[:len(smoothed_mean_heights_frame)//2].reshape(1, -1))[0])
        # Append to our list
        mean_heights.append(smoothed_mean_heights_frame[:len(smoothed_mean_heights_frame)//2])

        # Calculate the mean height of the specified area and convert to meters
        area = height_map[x_start:x_end, y_start:y_end]
        for i in range(area.shape[0]):
            for j in range(area.shape[1]):
                if np.isnan(area[i][j]):
                    area[i][j] = 0 
        mean_height = np.mean(area) * conversion_factor

        # Append to IMU data list
        imu_data.append(mean_height)
        smoothed_imu = uniform_filter(imu_data, size=5)
        print(len(smoothed_imu))
        input_data_rf = np.array(mean_heights)
        output_data_rf = np.array(smoothed_imu) 
        #model_rf.fit(input_data_rf, output_data_rf)

    # Create an array for the x-axis representing the frame numbers
    x_values = np.arange(ds.dims['count'])

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, smoothed_imu, label='Actual IMU')
    plt.plot(x_values[600:], predicted_imu_rf, label='Random Forest Prediction', linestyle='dashed')
    plt.legend()
    plt.show()
else:
    print("Insufficient data to train the Random Forest model.")
