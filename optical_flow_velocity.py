import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cv2


def calculate_velocity(height_maps, direction_angle):
    # Initialize a list to store the velocity arrays
    velocities = []

    # Calculate the optical flow between each pair of consecutive height maps
    for i in range(1, len(height_maps)):
        flow = cv2.calcOpticalFlowFarneback(height_maps[i-1], height_maps[i], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Incorporate the direction of the flow
        flow_direction = np.array([np.cos(direction_angle), np.sin(direction_angle)])
        flow = flow * flow_direction
        
        # Calculate velocity and convert from pixels/frame to pixels/second by dividing by dt
        dt = 1/15  # Time step in seconds
        velocity = np.sqrt(flow[...,0]**2 + flow[...,1]**2) / dt

        # Append the velocity array to the list
        velocities.append(velocity)

        # Print the mean velocity between the current pair of frames
        print(f'Mean velocity between frames {i-1} and {i}: {np.nanmean(velocity)} pixels/second')

    return velocities


# Your array of arrays
height_maps = []

ds = xr.open_dataset('D:/Downloads/wassfast_output (1).nc')

for t in range(ds.dims['count']):
    da = ds['Z'].isel(count=t)
    array = da.values
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if np.isnan(array[i][j]):
                array[i][j] = 0 
            
    height_maps.append(array)

# Convert the list of arrays to a numpy array
height_maps = np.array(height_maps)

# The direction angle of the flow in radians
direction_angle = np.deg2rad(18.06)  # Replace with the actual direction angle

# Calculate the velocity
velocity = calculate_velocity(height_maps, direction_angle)
