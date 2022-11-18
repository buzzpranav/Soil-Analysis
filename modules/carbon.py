import cv2
import numpy as np

depth = 5
root_image = "CLAHE_Soil.png"
img = cv2.imread(root_image, 1)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
avg_color_per_row = np.average(hsv, axis=0)
avg_colors = np.average(avg_color_per_row, axis=0)
    
hue = avg_colors[0]
sat = avg_colors[1]
val = avg_colors[2]
print("Hue: " + str(hue))
print("Sat: " + str(sat))
print("Brightness: " + str(val))
    
# Step 1: Calculate Percent Clay 
clayPercent = round((-0.0853*sat)+37.1,2)

# Step 2: Estimate SOC % 
SOC = round((0.05262*hue) + (0.11041*clayPercent) + -2.76983,2)

# Step 3: Estimate Bulk Density (g/cm3)
bulkDensity = round((0.0129651*clayPercent) + (0.0030006*sat) + 0.4401499,2)

# SOC to SOM conversion function 
def SOM(SOC):
	SOM = round(SOC * 1.72,2)
	if SOM > 10:
		SOM = 10
	return SOM

# SOC (%) to SOC (tC/ha)
def SOCWeight (depth, BD, SOC):
	depthC = depth*2.54
	socWeight = round((SOC*0.01) * ((BD*(depthC/100)*10000)),2)
	return socWeight

# SOC eq 1 ==> Original eq. Textures used to train include Fine sandy loam and Silt loam 
SOC1 = round((0.0772*hue) + 1.72, 2)
if SOC1 > 5.8:
	SOC1 = 5.8

# SOC eq 2 ==> Updated eq. Textures used to train include Silt clay loam and Silt loam.
try:
	SOC2 = round((0.05262*hue) + (0.11041*clayPercent) + -2.76983, 2)
except:
	SOC2 = round((0.05902*hue) + -0.04238, 2)

# SOC eq 3 = Average of eq 1 and eq 2 predictions 
SOC3 = round(((SOC1+SOC2)/2),2)

# BULK Density eq 1 ==> predict BD using clay and saturation inputs 
BD = round((0.0129651*clayPercent) + (0.0030006*sat) + 0.4401499,2)


print("Clay Percentage Level: ", clayPercent)
print("________________________________")
print("SOC (%) Prediction 1: ", SOC1)
print("SOM (%) Prediction 1: ", SOM(SOC1))
print("________________________________")
print("SOC (%) Prediction 2: ", SOC2)
print("SOM (%) Prediction 2: ", SOM(SOC2))
print("________________________________")
print("SOC (%) Prediction 3: ", SOC3)
print("SOM (%) Prediction 3: ", SOM(SOC3))
print("________________________________")
print("Bulk Density (g/cm3): ", BD)
print("SOC Prediction 3 (tC/ha): ", SOCWeight(depth, BD, SOC3))