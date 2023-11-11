import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import line

mini=-600
med=50
maxi=700

ang=110

# Open the dataset
ds = xr.open_dataset('wassfast_output.nc')  # Assuming the file name is 'wassfast_output.nc'

# Select a specific time instant (e.g., the first time instant)
da = ds['Z'].isel(count=5)  # Replace 'Z' with the name of the variable that represents the sea wave heights

# Convert the DataArray to a 2D numpy array
array = da.values
# Define the start and end coordinates of the line

for i in range(array.shape[0]):
    for j in range(array.shape[1]):
        if array[i][j] < 100:
            array[i][j] = mini  
        elif array[i][j] > 100 and  array[i][j] < 300:
            array[i][j] = med  
        elif array[i][j] > 300: 
            array[i][j] = maxi



def draw_line(image, angle, centerx, centery, draw):
    center = (centerx, centery)
    angle = np.radians(angle)

    # Calculate the displacements along x and y directions
    dx = np.cos(angle)
    dy = np.sin(angle)
    
    # Calculate the intersections of the line with the image boundaries
    t_values = [
        (0 - center[0]) / dx,  # Intersection with top boundary
        (image.shape[0] - center[0]) / dx,  # Intersection with bottom boundary
        (0 - center[1]) / dy,  # Intersection with left boundary
        (image.shape[1] - center[1]) / dy  # Intersection with right boundary
    ]

    # Sort the t values and take the middle two values
    t_values.sort()
    t1, t2 = t_values[1:3]

    # Calculate the start and end points of the line
    start_point = center + np.array([dx * t1, dy * t1])
    end_point = center + np.array([dx * t2, dy * t2])

    # Draw the line
    if draw==True:
        plt.plot([start_point[1], end_point[1]], [start_point[0], end_point[0]], color='red')

    # Get the coordinates of all the points on the line
    line_coords = line(int(start_point[0]), int(start_point[1]), int(end_point[0]), int(end_point[1]))

    # Ensure all coordinates are within the valid range of indices for the array
    line_coords = (np.clip(line_coords[0], 0, image.shape[0] - 1), np.clip(line_coords[1], 0, image.shape[1] - 1))

    # Now you can safely index into the array
    heights = image[line_coords]
    trace= findRegions(heights)

    return heights, line_coords, trace

def findRegions(heights):
    high=0
    low=0
    prev=0
    trace=[]
    for i in range(len(heights)):
        
        if heights[i]==mini and prev!=maxi and prev!=med:
            low += 1
            prev=heights[i]
            if i== (len(heights)-1):
                trace.append(-low)
        elif heights[i]==maxi and prev!=mini:
            high += 1
            prev=heights[i]
            if i== (len(heights)-1):
                trace.append(high)

        elif heights[i]==med and prev!=mini:
            high += 1
            prev=heights[i]
            if i== (len(heights)-1):
                trace.append(high)
        
        elif heights[i]==med and prev==mini:
            if (low)>=4:
                trace.append(-low)
            low=0
            high += 1
            prev=heights[i]

        elif heights[i]==maxi and prev==mini:
            if low >=4:
                trace.append(-low)
            low=0
            high += 1
            prev=heights[i]

        elif heights[i]==mini and prev!=mini:
            if high>=4:
                trace.append(high)
            high=0
            low += 1
            prev=heights[i]
        elif np.isnan(heights[i]):
           
            if prev==mini:
                trace.append(-low)
                low=0
            elif prev==med:
                trace.append(high)
                high=0
            elif prev == maxi:
                trace.append(high)
                high=0
            trace.append("nan")
            prev="nan"

        
        
    return trace

def orthogonal(array, heights, line_coords, ang):
    
    # Initialize your dictionary
    ort = []
    trace=findRegions(heights)
    numb = 0
    
    for j in range(len(trace)):
        if trace[j]=="nan":
            numb += 1
        elif trace[j] < 0:
            numb += abs(trace[j])
        elif trace[j] > 0:
            maxv=0
            summax=0
            # Search for the value with most highest points
            for i in range(trace[j]): #starts counting from the length of that part
                sum=0
                curv=int(numb + i)
                _, _, region = draw_line(array, (ang-90), line_coords[0][curv], line_coords[1][curv], False)
                for t in range(len(region)): #ads all positive numbers on the trace
                    if region[t]!="nan" and region[t]>0:
                        sum+=region[t]

                if sum>summax:
                    summax=sum
                    maxv=curv

            ort.append(maxv)

            numb = numb + trace[j]

    for t in range(len(ort)):
        draw_line(array, (ang-90), line_coords[0][ort[t]], line_coords[1][ort[t]], True)
        
    
    





# Call the function with your array and the desired angle (in degrees)
heights, line_coords, trace = draw_line(array, ang, 128, 128, True)  # Draws a line at 90 degrees
print(trace)

orthogonal(array, heights, line_coords, ang)










# Print the coordinates of all the points on the line
#print("heights of the line:", heights)

plt.imshow(array)
plt.colorbar(label='Sea wave height')

plt.show()




