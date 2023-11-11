import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import line

mini=-600
med=50
maxi=700

ang=0

# Open the dataset
ds = xr.open_dataset('wassfast_output.nc')  # Assuming the file name is 'wassfast_output.nc'

# Select a specific time instant (e.g., the first time instant)
da = ds['Z'].isel(count=0)  # Replace 'Z' with the name of the variable that represents the sea wave heights

# Convert the DataArray to a 2D numpy array
array = da.values
# Define the start and end coordinates of the line

def add_consecutive_same_sign(arr):
    if not arr:  # if the array is empty
        return arr

    result = [arr[0]]
    for num in arr[1:]:
        if num == 'nan':
            result.append(num)
            continue

        num = int(num)
        if result[-1] != 'nan' and ((result[-1] >= 0 and num >= 0) or (result[-1] < 0 and num < 0)):
            result[-1] += num
        else:
            result.append(num)

    return result



for i in range(array.shape[0]):
    for j in range(array.shape[1]):
        if array[i][j] < 120:
            array[i][j] = mini  
        elif array[i][j] > 120 and  array[i][j] < 300:
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
    rescue=0
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
            if (low)>=10:
                trace.append(-low)
            elif 0<low<10:
                high+=low
            low=0
            high += 1
            prev=heights[i]

        elif heights[i]==maxi and prev==mini:
            if low >=10:
                trace.append(-low)
            elif 0<low<10:
                high+=low
            low=0
            high += 1
            prev=heights[i]

        elif heights[i]==mini and prev!=mini:
            if high>=10:
                trace.append(high)
            elif 0<high<10:
                low+=high
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

def orthogonal(array):
    
    # Initialize your dictionary

    summax=0
    angmax=0
   
    
    
    for w in range(179):
        #print(w)
        
        numb=0
        
        _, line_coords, trace = draw_line(array, w, 128, 128, False)
        
        sumangmax=0
        tracemax=[]
        for j in range(len(trace)):#how many segments are there in a line e.g.["nan",23,-15,7,-30] It will do this 5 times
            if trace[j]=="nan":
                numb += 1
            elif trace[j] < 0:
                numb += abs(trace[j])
            elif trace[j] > 0: #if the segment is positive
                maxv=0
                
                # Search for the value with most highest points  
                sum=0
                curv=int(numb + int(trace[j]/2))
                _, _, region = draw_line(array, (w-90), line_coords[0][curv], line_coords[1][curv], False) #draws a perpendicular line at the middle point
                for t in range(len(region)): #ads all positive numbers on the trace
                    if region[t]!="nan" and region[t]>0:
                        sum+=region[t]


                sumangmax+=sum
                
                maxv=curv
                    
                

                numb = numb + trace[j]
            #sumang+=summax
        if sumangmax>summax:
            summax=sumangmax
            angmax=w
            print(summax, angmax)
            


        #if angmax==w:
                
    
    


    curort = []

    numb = 0
    _, line_coords, tracemax = draw_line(array, angmax, 128, 128, False)
    print(tracemax)
    tracemax=add_consecutive_same_sign(tracemax)  
    print(tracemax)
    for j in range(len(tracemax)):
        if tracemax[j]=="nan":
            numb += 1
        elif tracemax[j] < 0:
            
            numb += abs(tracemax[j])
        elif tracemax[j] > 0:
            maxv=0
            summax=0
            # Search for the value with most highest points
            for i in range(tracemax[j]): #starts counting from the length of that part
                sum=0
                curv=int(numb + i)
                _, _, region = draw_line(array, (angmax-90), line_coords[0][curv], line_coords[1][curv], False)
                for t in range(len(region)): #ads all positive numbers on the trace
                    if region[t]!="nan" and region[t]>0:
                        sum+=region[t]
                        
                if sum>summax:
                    summax=sum
                    maxv=curv
                

            curort.append(maxv)
            numb = numb + tracemax[j]
                
                
        
        
        
        
    draw_line(array, angmax, 128, 128, True)
    for t in range(len(curort)):
        draw_line(array, (angmax-90), line_coords[0][curort[t]], line_coords[1][curort[t]], True)
        
    
    






orthogonal(array)










# Print the coordinates of all the points on the line
#print("heights of the line:", heights)

plt.imshow(array)
plt.colorbar(label='Sea wave height')

plt.show()




