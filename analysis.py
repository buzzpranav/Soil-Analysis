'''
SOIL ANALYSIS

Author(s): Pranav S Narayanan, Sudeshna Das
Copyright (c) [2022] [Pranav S Narayanan]
License: MIT License

This program uses a picture of soil to quickly and accurately find soil composition and relevant 
information such as pH Value, Clay Percentage, Bulk Density, Organic Carbon Levels, 
Organic Matter Levels, Electrical Conductivity of the soil. It then uses this 
information to create a detailed yet easy to understand report which farmers can use to plant the 
right crops and adequete fertilizers. For the full introduction, checkout the README.md
'''

#&--Precedure:
#*Step 0: Setup libraries and settings
#*Step 1: Preprocess Image using CLAHE
#*Step 2: Split original image into various color spaces
#*Step 3: Convert image into various color spaces
#*Step 4: Find pH using Random Forest algorithm
#*Step 5: Find Organic Carbon using HSV and Clay properties, then use SOC to find Organic Matter level
#*Step 6: Using another Random Forest Algorithm, find Electrical conductivity
#*Step 7: Using soil type and other collected information, suggest crops

#~~Step 0.0: Import Required Libraries
import pickle
import numpy as np
import cv2    
import os
import pandas as pd
import colorgram
import matplotlib.pyplot as plt
from matplotlib import colors
import joblib
from colorama import Fore, Back, Style, init
init(autoreset = True)
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from sklearn.preprocessing import StandardScaler
from datetime import datetime
now = datetime.now()

#~~ Step 0.1: Setup Analysis Settings
pd.options.mode.chained_assignment = None
soil_analysis = True
detailed_analysis = False
show_graph = False

#~~Step 1: CLAHE (Contrast Limiting Adaptive Histogram Enhancement) Algorithm
def ImgEnhancer(original_image):
    img = cv2.imread(original_image)
   
    #~~Step 1.1: Convert Color space to LAB (Lightness, A chromaticity, B chromaticity)
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
 
    #~~Step 1.2: Split color channels
    l, a, b = cv2.split(lab_img)
 
    #~~Step 1.3: Apply CLAHE to the Lightness space (L-channel)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
    clahe_lab_img = clahe.apply(l)
 
    #~~ Step 1.4: Combine the CLAHE enhanced L-channel back to A and B channels
    updated_clahe_lab_img = cv2.merge((clahe_lab_img, a, b))
 
    #~~ Step 1.5: Convert color space back to BGR
    final_img = cv2.cvtColor(updated_clahe_lab_img, cv2.COLOR_LAB2BGR)
 
    #~~ Step 1.6: Scale and save image to uploads folder
    img_scaled = cv2.resize(final_img, (400, 400), interpolation=cv2.INTER_AREA)
    pre_process = "CLAHE_" + os.path.basename(original_image)
    cv2.imwrite(pre_process, img_scaled)
 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#~~ Step 2: Convert image to RGB, HSV, Lab, and XYZ spaces
def ColorSpace(root_image):
    img = cv2.imread(root_image, 1)

    #~~ Step 2.1: Convert image from BGR (24-bit Blue, Green, Red) to RGB (256 Red, Green Blue)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r, g, b = cv2.split(rgb_img)

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")

    pixel_colors = rgb_img.reshape((np.shape(rgb_img)[0]*np.shape(rgb_img)[1], 3))
    norm = colors.Normalize(vmin=-1.0, vmax=1.0)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    axis.scatter(r.flatten(), g.flatten(), b.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Red")
    axis.set_ylabel("Green")
    axis.set_zlabel("Blue") 
    if show_graph == True:
        plt.show()
    rgbFileName = "graphs/" + "colorSRGB_" + os.path.basename(root_image)
    plt.savefig(rgbFileName)

    def getAverageRGBN(rgb_img):
        im = np.array(rgb_img)
        w,h,d = im.shape
        im.shape = (w*h, d)
        return tuple(im.mean(axis=0))
    
    if detailed_analysis == True:
        print('Average RGB values are',getAverageRGBN(rgb_img))

    #~~ Step 2.2: Convert image from RGB to HSV (Hue, Saturation, Value)
    hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_img)

    pixel_colors = hsv_img.reshape((np.shape(hsv_img)[0]*np.shape(hsv_img)[1], 3))
    norm = colors.Normalize(vmin=-1.0, vmax=1.0)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")
    if show_graph == True:
        plt.show()
    hsvFileName = "graphs/" + "colorSHSV_" + os.path.basename(root_image)
    plt.savefig(hsvFileName)
    def getAverageHSVN(hsv_img):
        im = np.array(hsv_img)
        w,h,d = im.shape
        im.shape = (w*h, d)
        return tuple(im.mean(axis=0))
    
    if detailed_analysis == True:
        print('Average HSV values are',getAverageHSVN(hsv_img)) 

    #~~ Step 2.3: Convert image from RGB to Lab (Lightness, Chromaticity)
    lab_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2Lab)
    l, a, b = cv2.split(lab_img)

    pixel_colors = lab_img.reshape((np.shape(lab_img)[0]*np.shape(lab_img)[1], 3))
    norm = colors.Normalize(vmin=-1.0, vmax=1.0)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("Light*")
    axis.set_ylabel("a* value")
    axis.set_zlabel("b* Value")
    if show_graph == True:
        plt.show()
    labFileName = "graphs/" + "colorSLAB_" + os.path.basename(root_image)
    plt.savefig(labFileName)
    def getAverageLabN(lab_img):
        im = np.array(lab_img)
        w,h,d = im.shape
        im.shape = (w*h, d)
        return tuple(im.mean(axis=0))

    if detailed_analysis == True:
        print('Average Lab values are',getAverageLabN(lab_img))

    #~~Step 2.4: convert image from RGB to XYZ (Average Human Color Spectrum)
    XYZ_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2XYZ)
    X, Y, Z = cv2.split(XYZ_img)

    pixel_colors = XYZ_img.reshape((np.shape(XYZ_img)[0]*np.shape(XYZ_img)[1], 3))
    norm = colors.Normalize(vmin=-1.0, vmax=1.0)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    axis.scatter(h.flatten(), s.flatten(), v.flatten(), facecolors=pixel_colors, marker=".")
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    if show_graph == True:
        plt.show()
    xyzFileName = "graphs/" + "colorSXYZ_" + os.path.basename(root_image)
    plt.savefig(xyzFileName)
    def getAverageXYZN(XYZ_img):
        im = np.array(XYZ_img)
        w,h,d = im.shape
        im.shape = (w*h, d)
        return tuple(im.mean(axis=0))
    
    if detailed_analysis == True:
        print('Average XYZ values are',getAverageXYZN(XYZ_img)) 

def is_soil(root_image):
    main_color = colorgram.extract(root_image, 3)
    rgb = main_color[0].rgb
    main_r,main_g,main_b = rgb[0],rgb[1],rgb[2]
    if (main_r > 25 and main_r < 190) and (main_g > 10 and main_g < 180) and (main_b > 10 and main_b < 120):
        return True
    else:
        return False

#~~ Step 3: Find Soil pH Value Using Random Forest Model
def pH(root_image):
    img = cv2.imread(root_image, 1)

    #~~ Step 3.1: Convert image from BGR to RGB and split channels
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    im = np.array(rgb_img)
    w,h,d = im.shape
    im.shape = (w*h, d)
    rgb_list = list(im.mean(axis=0))

    #~~ Step 3.2: Read rgb-ph data and appened rgb values from Step 5.1
    ds = pd.read_csv('data\pH_data.csv')
    X = ds.iloc[:, :-1]
    X.loc[len(X)] = rgb_list
    X = X.values

    sc = StandardScaler()
    X = sc.fit_transform(X)

    #~~ Step 3.3: Load Pretrained Random Forest model and predict last appended value
    model = joblib.load('models\pH_model.pkl')
    pred = model.predict(X)[-1]
    ph = (pred * 0.65) + 0.5

    print(Style.BRIGHT + Fore.BLUE + " Your soil's pH value is:", round(float(ph),2))
    if int(ph) < 5.5:
        print(Style.BRIGHT + Fore.YELLOW + Back.RED + " The pH value is too low! Please consider adding lime subtances to your soil.")
    elif int(ph) > 7.5:
        print(Style.BRIGHT + Fore.YELLOW + Back.RED + " The pH value is too high! Pleace consider adding sulphuric compoounds to your soil.")
    else:
        print(Style.BRIGHT + Fore.WHITE + Back.GREEN + " The pH value is perfect for gardening and agriculture!")

#~~ Step 4: Find Organic Material Level
def Organic_Material(root_image):
    img = cv2.imread(root_image, 1)

    #~~ Step 4.1: Convert to HSV and extract hsv values
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    avg_color_per_row = np.average(hsv, axis=0)
    avg_colors = np.average(avg_color_per_row, axis=0)
        
    hue = avg_colors[0]
    sat = avg_colors[1]
    val = avg_colors[2]
        
    #~~ Step 4.2: Calculate clay percentage using saturation 
    clayPercent = round((-0.0853*sat)+37.1,2)


    #~~ Step 4.3: Find organic Carbon level using fine sandy loam and Silt loam formula 
    SOC_SandySiltLoam = round((0.0772*hue) + 1.72, 2)
    if SOC_SandySiltLoam > 5.8:
        SOC_SandySiltLoam = 5.8

    #~~ Step 4.4: Find organic Carbon level using silt clay loam and Silt loam formula 
    try:
        SOC_ClaySiltLoam = round((0.05262*hue) + (0.11041*clayPercent) + -2.76983, 2)
    except:
        SOC_ClaySiltLoam = round((0.05902*hue) + -0.04238, 2)

    #~~ Step 4.5: Find final organic carbon level (= Average of eq 1 and 2)
    Organic_Carbon = round(((SOC_SandySiltLoam+SOC_ClaySiltLoam)/2),2)

    #~~ Step 4.6: Find Bulk Density using clay and saturation inputs 
    BD = round(((0.0129651*clayPercent) + (0.0030006*sat) + 0.4401499) * 1.4, 2)

    #~~ Step 4.7 Convert Organic Carbon level to Organic Matter level 
    def SOM(SOC):
        SOM = round(SOC * 1.72,2)
        if SOM > 10:
            SOM = 10
        return SOM

    print(Style.BRIGHT + Fore.BLUE + " The soil's clay profile is: ", clayPercent, "%")
    if int(clayPercent) > 40:
        print(Style.BRIGHT + Fore.YELLOW + Back.RED + " The clay percentage is too high! Consider adding organic compost to your soil.")
    elif int(clayPercent) < 15:
        print(Style.BRIGHT + Fore.YELLOW + Back.RED + " The clay percentage is too low! Consider adding more clay to your soil.")
    else:
        print(Style.BRIGHT + Fore.WHITE + Back.GREEN + " The Soil Contains Suitable Amounts Of Clay For Agriulture and Gardening!")
    
    print("")

    print(Style.BRIGHT + Fore.BLUE + " The Soil's Bulk Density (g/cm3) is: ", BD)
    if int(BD) > 1.75:
        print(Style.BRIGHT + Fore.YELLOW + Back.RED + " The Bulk Density is too high! Please try to compress your soil for better agriculture.")
    elif int(BD) < 1.5:
        print(Style.BRIGHT + Fore.YELLOW + Back.RED + " The Bulk Density is too low! Please utilize multi-crop systems and break up compacted soil layers.")
    else:
        print(Style.BRIGHT + Fore.WHITE + Back.GREEN + " The Soil Contains The Perfect Amount Of Bulk Density For Agriulture and Gardening!")
    
    print("")

    print(Style.BRIGHT + Fore.BLUE + " The Soil's Organic Carbon Level is: ", Organic_Carbon, "%")
    if int(Organic_Carbon) > 4:
        print(Style.BRIGHT + Fore.YELLOW + Back.RED + " The Organic Carbon Levels are too high! Consider adding organic compost to your soil.")
    elif int(Organic_Carbon) < 0.7:
        print(Style.BRIGHT + Fore.YELLOW + Back.RED + " The Organic Carbon Levels are too low! Consider adding more clay to your soil.")
    else:
        print(Style.BRIGHT + Fore.WHITE + Back.GREEN + " The Soil Contains The Perfect Amount Of Organic Carbon Levels For Agriulture and Gardening!")

    print("")

    print(Style.BRIGHT + Fore.BLUE + "The Soil's Organic Matter Level is: ", SOM(Organic_Carbon), "%")
    if float(SOM(Organic_Carbon)) > 3.25:
        print(Style.BRIGHT + Fore.YELLOW + Back.RED + "The Organic Matter Levels are too high! Consider adding organic compost to your soil.")
    elif float(SOM(Organic_Carbon)) < 2.75:
        print(Style.BRIGHT + Fore.YELLOW + Back.RED + "The Organic Matter Levels are too low! Consider adding more clay to your soil.")
    else:
        print(Style.BRIGHT + Fore.WHITE + Back.GREEN + "The Soil Contains The Perfect Amount Of Organic Matter Levels For Agriulture and Gardening!")

#~~ Step 5: Find Electrical Conductivity using Random Forest Model
def Electrical_Conductivity(root_image):
    image = cv2.imread(root_image)
    
    #~~ Step 5.1: Extract blue, red, green channels from image
    blue_channel = image[:,:,0]
    green_channel = image[:,:,1]
    red_channel = image[:,:,2]
    temp = ((np.median(green_channel)+np.median(blue_channel))+np.median(red_channel))
    temp = np.nanmean(temp)

    #~~ Step 5.2: Load Pretrained Models
    ECmodelclass = pickle.load(open('models\ECclassifier.pkl', 'rb'))

    #~~ Step 5.3: Predict EC using pretrained models
    ECresult = float(ECmodelclass.predict([[temp]]))
    print(Style.BRIGHT + Fore.BLUE + " The Electricial Conductivity of this soil is:", round(ECresult,3))
    if float(ECresult) > 0.57:
        print(Style.BRIGHT + Fore.YELLOW + Back.RED + " The Electrical Conductivity is too high! Consider using better irrigation practices and avoid waterlogging")
    elif float(ECresult) < 0.15:
        print(Style.BRIGHT + Fore.YELLOW + Back.RED + " The Electrical Conductivity is too low! Consider adding organic matter, such as manure and compost to your soil")
    else:
        print(Style.BRIGHT + Fore.WHITE + Back.GREEN + " The Soil Contains The Perfect Amount Of Salt & Electrical Conductivity For Agriulture and Gardening!")

#~~ Step 7: Compile and Run Together
if soil_analysis == True:
    print(Style.BRIGHT + Fore.WHITE + " Soil Analysis at: " + now.strftime("%d-%m-%Y %H:%M"))
    print(Style.BRIGHT + Fore.MAGENTA + " ---------------------------------------------")
    original_image = input(Style.BRIGHT + Fore.CYAN + " Enter Your Image File Location: ")
    if original_image == "":
        original_image = ("LoamySoil.png")
    print("")
    soil_confirmation = is_soil(original_image)
    if soil_confirmation==True:
        ImgEnhancer(original_image)
        root_image = ("CLAHE_" + original_image)
        if detailed_analysis == True:
            ColorSpace(root_image)
        pH(root_image)
        print("")
        Organic_Material(root_image)
        print("")
        Electrical_Conductivity(root_image)
        print("")
    else:
        print("This Does Not Seem Like Soil! Please Try Again.")