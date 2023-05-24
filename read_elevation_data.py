from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

### DATA INFO:
# Image resolution: every pixel is 10 meters
# Land location in np array: (600, 1560)
# Takeoff location in np array: (2015, 930)

### PROCEDURE FOR LOADING DEM FILE TO NP ARRAY
# Load image from DEM (digital elevation map) file (.tif)
im = Image.open("dtm10_7002_2_10m_z33.tif")

# Reshape to only contain relevant area
w, h = im.size
im = im.crop((0, 0, w / 2, h / 2))

# Load into numpy array
imarray = np.array(im)
print("Land:",imarray[600, 1560])
print("Take off:",imarray[2015, 930])
print("Land2:",imarray[1560,600])
print("Take off2:",imarray[930,2015])
# array_3d = np.zeros((2560, 2560, 120))
# for z in range(0,120):
#     for x in range(0,imarray.shape[0]):
#         for y in range(0,imarray.shape[1]):
#             if imarray[x,y]>=z:
#               array_3d[x,y,z]=1
            

# Plot data
plt.imshow(imarray)
plt.scatter(600, 1560, c="red", label="land")
plt.scatter(2015, 930, c="green", label="takeoff")
plt.legend()
plt.colorbar()
plt.title("Height map of Trondheim area (10m resolution)")
plt.show()
