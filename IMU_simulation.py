import xarray as xr
import numpy as np
from scipy.ndimage import uniform_filter

# Load your netCDF file
ds = xr.open_dataset('/content/wassfast_output.nc')

# Define the area of interest
x_start, x_end = 100, 150
y_start, y_end = 100, 150

# Initialize an empty list to hold the mean heights
mean_heights = []

# Conversion factor from original units to meters
conversion_factor = 0.6 / 600

# Loop over each time frame
for t in range(ds.dims['count']):
    # Select the area of interest
    da = ds['Z'].isel(count=t)
    area = da.values[x_start:x_end, y_start:y_end]
    for i in range(area.shape[0]):
      for j in range(area.shape[1]):
          if np.isnan(area[i][j]):
              area[i][j] = 0 
    
    
    # Calculate the mean height and convert to meters
    mean_height = np.mean(area) * conversion_factor
    
    # Append to our list
    mean_heights.append(mean_height)

# Convert to a numpy array for further processing
mean_heights = np.array(mean_heights)

# Smooth the curve using a uniform filter
smoothed_mean_heights = uniform_filter(mean_heights, size=20)


# Calculate velocity and acceleration
dt = 1/15  # time step in seconds
velocity = np.gradient(smoothed_mean_heights, dt)
acceleration = np.gradient(velocity, dt)

# Now, 'acceleration' holds your acceleration values
print("positions: ",smoothed_mean_heights)
print("velocities: ",velocity)
print("accelerations: ",acceleration)
