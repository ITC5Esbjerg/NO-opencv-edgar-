import numpy as np
import pandas as pd
import pickle
from statsmodels.tsa.ar_model import AutoReg
import xarray as xr
from skimage.draw import line
from scipy.interpolate import make_interp_spline, BSpline
from scipy.ndimage import uniform_filter
x_start, x_end = 50, 100
y_start, y_end = 130, 180
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
ds = xr.open_dataset('D:/Downloads/wassfast_output.nc')

# Initialize an empty list to hold the mean heights
mean_heights = []
imu_data = []

# Conversion factor from original units to meters
conversion_factor = 0.6 / 600

# Loop over each time frame
for t in range(ds.dims['count']):
    # Select the entire frame
    da = ds['Z'].isel(count=t)
    height_map = da.values
    height_map = np.nan_to_num(height_map, nan=0)

    heights, line_coords = draw_line(height_map, 109, 128, 128)
    mean_heights_frame = []
    for k in range(len(line_coords[0])):
        orthogonal_heights, _ = draw_line(height_map, 109+90, line_coords[0][k], line_coords[1][k])
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
# Convert to a numpy array for further processing

input_data = np.array(mean_heights)
output_data = np.array(smoothed_imu)  # This is your simulated IMU position

# Train the AutoReg model
model = AutoReg(output_data, lags=1)
model_fit = model.fit()

# Save the model to a pickle file
with open('model.pkl', 'wb') as f:
    pickle.dump(model_fit, f)
