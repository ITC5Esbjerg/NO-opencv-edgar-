import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cv2

mini=0
med=200
maxi=700
val=10

def draw_arrow(image, angle, centerx, centery, length, draw):
    center = np.array([centerx, centery])
 

    dx = np.cos(angle)
    dy = np.sin(angle)

    direction = np.array([dx, dy])

    start_point = center - length / 2 * direction
    end_point = center + length / 2 * direction

    if draw==True:
        plt.arrow(start_point[1], start_point[0], end_point[1]-start_point[1], end_point[0]-start_point[0], color='red', width=1, head_width=10, head_length=10)

    

def calculate_velocity(height_maps, direction_angle):
    velocities = []
    for i in range(1, len(height_maps)):
        flow = cv2.calcOpticalFlowFarneback(height_maps[i-1], height_maps[i], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Define the flow direction
        flow_direction = np.array([np.cos(direction_angle), np.sin(direction_angle)])
        
        # Calculate the angle of each vector
        flow_angle = np.arctan2(flow[..., 1], flow[..., 0])
        
        # Create a mask for vectors that are within +-20 degrees from the arrow
        threshold = np.radians(20)  # Convert the threshold to radians
        mask = np.abs(flow_angle - direction_angle) < threshold
        
        # Apply the mask to the flow array when calculating flow_in_direction
        flow_in_direction = np.sum(flow[mask] * flow_direction, axis=-1)
        
        dt = 1/7.5
        velocity = flow_in_direction / dt
        scale_factor = 50 / 256
        velocity = velocity * scale_factor
        velocities.append(velocity)
        print(f'Mean velocity between frames {i-1} and {i}: {np.nanmean(velocity)} m/s')
        
        # Plot the final vector map
        plt.figure(figsize=(10, 10))
        plt.quiver(np.arange(0, flow.shape[1], 10), np.arange(0, flow.shape[0], 10), flow[::10, ::10, 0], flow[::10, ::10, 1], color='r')
        
        # Draw a line at the given angle through the center of the image
        draw_arrow(flow, direction_angle, flow.shape[0] // 2, flow.shape[1] // 2, 50, True)
        
        # Set the limits of the axes
        plt.xlim(0, flow.shape[1])
        plt.ylim(flow.shape[0], 0)  # Note: The y-axis is inverted in image coordinates
        
        
    
    return velocities


height_maps = []
ds = xr.open_dataset('D:/Downloads/wassfast_output.nc')

for t in range(0,50,2):
    da = ds['Z'].isel(count=t)
    array = da.values
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if np.isnan(array[i][j]):
                array[i][j] = -600 
        
            
    height_maps.append(array)

height_maps = np.array(height_maps)
direction_angle = np.deg2rad(109)
velocity = calculate_velocity(height_maps,direction_angle)
