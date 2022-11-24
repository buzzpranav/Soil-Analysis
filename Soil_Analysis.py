'''
SOIL ANALYSIS

Author: Pranav S Narayanan
Copyright (c) [2022] [Pranav S Narayanan]
License: MIT License

This program uses a picture of soil to quickly and accurately find soil composition and relevant 
information such as pH Value, Soil Roughness, Clay Percentage, Bulk Density, Organic Carbon Levels, 
Organic Matter Levels, Phosphorus quantities, Electrical Conductivity of the soil. It then uses this 
information to create a detailed yet easy to understand report which farmers can use to plant the 
right crops and adequete fertilizers. For the full introduction, checkout the README.md
'''

#&--Backend Precedure:
#*Step 0: Setup libraries and settings
#*Step 1: Preprocess Image using CLAHE on ROI
#*Step 2: Convert image to array of RGB values
#*Step 3: Convert image into various color spaces
#*Step 4: Find pH using classical formula
#*Step 5: Find pH using Random Forest algorithm
#*Step 6: Fetch soil texture and roughness with magic
#*Step 7: Find Organic Carbon and Organic Matter level via HSV and Clay properties
#*Step 8: Run pretrained CNN algorithms to get pH, Phosphorus, Organic Matter, & Electrical Conductivity level
#*Step 9: Compile steps 1-8 into single function which can be fed into the frontend
#TODO 1: Find soil moisture from RGB image
#TODO 2: Convert geolocation to hyperspectral images for more accuracy (experimental)
#TODO 3: Find soil elemental composition (nitrogen, phosphorus, sulphur, etc)
#TODO 4: Use classification model with solved parameters as input to predict best crops and fertilizers

#&--Frontend Procedure:
#TODO 1: Finalize framework (Flask, Django, Kivy, Android Studio, PyQT5)
#TODO 2: Input image as file upload / camera shot
#TODO 3: Attempt to Autofetch geolocation
#TODO 4: Send image to backend function to receive analysis info
#TODO 5: Use geolocation to fetch hyperspectral image for additional info (Experimental)
#TODO 6: Convert info to analytical report w/ ideal levels for each parameter
#TODO 7: Use info from analytical report to determine ideal crops and fertlizers

#~~Step 0.0: Import Required Libraries
import pickle
import numpy as np
import cv2    
import os
import pandas as pd
import matplotlib.pyplot as plt
from kneed import KneeLocator
from collections import Counter
from matplotlib.colors import to_hex
from matplotlib import colors
import joblib
import math
import PIL
import warnings
from imageio import imread
from skimage.transform import resize
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#~~ Step 0.1: Setup Analysis Settings
pd.options.mode.chained_assignment = None
soil_analysis = True
detailed_analysis = False
show_graph = False
langauge = "English"
location = "India"

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

    #~~ Step 1.5: Convert color space bacl to BGR
    final_img = cv2.cvtColor(updated_clahe_lab_img, cv2.COLOR_LAB2BGR)

    #~~ Step 1.6: Scale and save image to uploads folder
    img_scaled = cv2.resize(final_img, (400, 400), interpolation=cv2.INTER_AREA)
    pre_process = "CLAHE_" + os.path.basename(original_image)
    cv2.imwrite(pre_process, img_scaled)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

#~~Step 2: Read image file as 2-D array of RGB values
def ColorPalette(root_image):
    imgName = root_image

    #~~Step 2.1: Read and resize image
    img = imread(imgName)
    img = resize(img, (400, 400))

    #~~Step 2.2: Created a (400,400,3) array of RGB values of the image 
    data = pd.DataFrame(img.reshape(-1, 3),
                        columns=['R', 'G', 'B'])

    #~~Step 2.3: Replace zero and infinity with Nan
    df = pd.DataFrame(data)
    df[:] = np.nan_to_num(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    #~~Step 2.4: Define within cluster sum square error (wcsse) and vary the number of clusters
    wcsse = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=23)
        kmeans.fit(df)
        wcsse.append(kmeans.inertia_)

    #~~Step 2.5: Plot and generate Elbow curve (to chart all possible cluster counts in K-means)
    plt.figure(figsize=(10, 8))
    plt.plot(range(1, 11), wcsse, marker='o', linestyle='--')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSSE')
    plt.title('Elbow Curve')
    if show_graph == True:
        plt.show()

    #~~Step 2.6: Find d[aakae]Elbow-point (optimal cluster count)
    kl = KneeLocator(range(1, 11), wcsse, curve="convex", direction="decreasing")
    k = kl.elbow
    
    #~~Step 2.7: Initialize K-means clustering based on the Elbow value
    np.random.seed(0)
    kmeans = KMeans(n_clusters=k, init='k-means++',
                    random_state=0)

    #~~Step 2.8: Fit and assign clusters
    data['Cluster'] = kmeans.fit_predict(data)

    #~~Step 2.9: set counts from cluster data
    counts = Counter(data['Cluster'])
    center_colors = kmeans.cluster_centers_

    #~~Step 2.10: Order colors by iterating through the counts
    ordered_colors = [center_colors[i] for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]
    plt.figure(figsize=(8, 6))
    plt.pie(counts.values(), colors=rgb_colors)
    if show_graph == True: 
        plt.show()

    #~~Step 2.11: Get the color palette from the cluster centers
    palette = kmeans.cluster_centers_
    palette_list = list()

    #~~Step 2.12: Convert data to format accepted by matplotlib
    for color in palette:
        palette_list.append([[tuple(color)]])

    #~~ Step 2.13: Show color palette
    for color in palette_list:
        plt.figure(figsize=(1, 1))
        plt.axis('off')
        plt.imshow(color)

    #~~ Step 2.14: Recreate the image using only colors from color palette with cluster centroid
    data['R_cluster'] = data['Cluster'].apply(lambda x: palette_list[x][0][0][0])
    data['G_cluster'] = data['Cluster'].apply(lambda x: palette_list[x][0][0][1])
    data['B_cluster'] = data['Cluster'].apply(lambda x: palette_list[x][0][0][2])

    #~~ Step 2.15: Convert the dataframe back to a numpy array
    img_c = data[['R_cluster', 'G_cluster', 'B_cluster']].values

    #~~ Step 2.16: Reshape the data back to a 400x400 image
    img_c = img_c.reshape(400, 400, 3)

    #~~ Step 2.17: Display the image
    plt.axis('off')
    plt.imshow(img_c)
    if show_graph == True: 
        plt.show()
    pre_process = "graphs/" + "colorP_" + os.path.basename(root_image)
    plt.imsave(pre_process, img_c)

#~~ Step 3: Convert image to RGB, HSV, Lab, and XYZ spaces
def ColorSpace(root_image):
    img = cv2.imread(root_image, 1)

    #~~ Step 3.1: Convert image from BGR (24-bit Blue, Green, Red) to RGB (256 Red, Green Blue)
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

    #~~ Step 3.2: Convert image from RGB to HSV (Hue, Saturation, Value)
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

    #~~ Step 3.3: Convert image from RGB to Lab (Lightness, Chromaticity)
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

    #~~Step 3.4: convert image from RGB to XYZ (Average Human Color Spectrum)
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

#~~ Step 4: Find soil pH value using classical formula
def classical_ph(root_image):

    #~~ Step 4.1: Read Data
    df=pd.read_csv('data\classic_pH_data.csv')
    g_frac=np.array(df.g_fraction)      
    pH=np.array(df.pH)               
    soil_data=[]                      

    for i in range(len(pH)):
        sd=[g_frac[i], pH[i]]
        soil_data.append(sd)

    #~~ Step 4.2: Calculate Average RGB Values
    imgName = root_image 
    image = PIL.Image.open(imgName)  
    image = image.resize((400,400))
    image_rgb = image.convert("RGB")

    i=0
    j=0
    c=[]
    for i in range(400):
        for j in range(400):
            c.append([i,j])
            j+=1
        i+=1

    RGB=[]
    for i in c:
        rgb_value = image_rgb.getpixel((i[0],i[1]))
        RGB.append(rgb_value)
        
    R=[]
    G=[]
    B=[]
    for i in RGB:
        R.append(i[0])
        G.append(i[1])
        B.append(i[2])
    R_mean=np.mean(R)
    G_mean=np.mean(G)
    B_mean=np.mean(B)

    #~~ Step 4.3: Use G-Fraction to find final pH
    g_frac_img=G_mean/(R_mean + G_mean + B_mean) 
    Data=[]           
    Dif=[]            
    for i in soil_data:
        g_frac_dif = np.abs(i[0]-g_frac_img)
        Data.append([g_frac_dif, i[1]])
        Dif.append(g_frac_dif)

    for j in Data:
        if j[0]==min(Dif):
            print("pH value from classical rgb prediction is:", j[1])

#~~ Step 5: Find soil pH value using Random Forest model
def Modern_pH(root_image):
    img = cv2.imread(root_image, 1)

    #~~ Step 5.1: Convert image from BGR to RGB and split channels
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    im = np.array(rgb_img)
    w,h,d = im.shape
    im.shape = (w*h, d)
    rgb_list = list(im.mean(axis=0))

    Red_val = rgb_list[0]
    Green_val = rgb_list[1]
    Blue_val = rgb_list[2]

    #~~ Step 5.2: Read rgb-ph data and appened rgb values from Step 5.1
    ds = pd.read_csv('data\modern_pH_data.csv')
    X = ds.iloc[:, :-1]
    X.loc[len(X)] = rgb_list
    X = X.values

    sc = StandardScaler()
    X = sc.fit_transform(X)

    #~~ Step 5.3: Load Pretrained Random Forest model and predict last appended value
    model = joblib.load('models\modern_pH_model.pkl')
    pred = model.predict(X)[-1]
    print("The Modern pH reading is:", pred)
    
#~~ Step 6: Record the texture of the image
def texture(root_image):

    #~~ Step 6.1: Don't touch this function. It works and no-one knows why.
    imgName = root_image
    img = cv2.imread(imgName,0) #import image
    img = cv2.resize(img, (400, 400))
    ft2 = np.fft.fft2(img)
    ftshift = np.fft.fftshift(ft2)
    w=img.shape[1] 
    h=img.shape[0] 

    magnitude_spectrum = 20*np.log(np.abs(ftshift))

    plt.axis('off')
    #plt.imshow(magnitude_spectrum, cmap = 'jet')
    # plt.savefig('uploads\colorTD_' + imgName)
    #RAPSD

    # Round up the size along this axis to an even number
    n = int( math.ceil(img.shape[0] / 2.) * 2 )

    # We use rfft since we are processing real values
    a = np.fft.rfft(img,n, axis=0)

    a = a.real*a.real + a.imag*a.imag
    ax = a.sum(axis=1)/a.shape[1]         #Amplitudes on x axis


    fx = np.fft.rfftfreq(n)   #frequencies on x axis     

    n = int( math.ceil(img.shape[1] / 2.) * 2 )

    a = np.fft.rfft(img,n,axis=1)

    a = a.real*a.real + a.imag*a.imag
    ay = a.sum(axis=0)/a.shape[0]      #Amplitudes on y axis

    fy = np.fft.rfftfreq(n)         #frequencies on y axis 


    r=(np.sqrt(ax**2 + ay**2))     #Amplitudes in radial direction


    plt.scatter(np.log(fx[1:]),np.log(r[1:]), s=5)
    m, c = np.polyfit(np.log(fx[1:]), np.log(r[1:]), 1)
    

    plt.plot(np.log(fx[1:]), m*np.log(fx[1:]) + c, 'r', label='y={:.2f}x+{:.2f}'.format(m,c))

    plt.legend()
    save_name = "graphs/" + "colorT_" +  os.path.basename(root_image)
    plt.savefig(save_name)
    if show_graph == True:
        plt.show()
    print('slope of line', round(m,3))
    h=(-m-1)/2
    print('surface roughness', round(h,3))

#~~ Step 7: Find Organic Carbon Level
def Organic_Carbon(root_image):
    depth = 5
    img = cv2.imread(root_image, 1)

    #~~ Step 7.1: Convert to HSV and extract hsv values
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    avg_color_per_row = np.average(hsv, axis=0)
    avg_colors = np.average(avg_color_per_row, axis=0)
        
    hue = avg_colors[0]
    sat = avg_colors[1]
    val = avg_colors[2]
        
    #~~ Step 7.2: Calculate clay percentage using saturation 
    clayPercent = round((-0.0853*sat)+37.1,2)


    #~~ Step 7.3: Find organic Carbon level using fine sandy loam and Silt loam formula 
    SOC_SandySiltLoam = round((0.0772*hue) + 1.72, 2)
    if SOC_SandySiltLoam > 5.8:
        SOC_SandySiltLoam = 5.8

    #~~ Step 7.4: Find organic Carbon level using silt clay loam and Silt loam formula 
    try:
        SOC_ClaySiltLoam = round((0.05262*hue) + (0.11041*clayPercent) + -2.76983, 2)
    except:
        SOC_ClaySiltLoam = round((0.05902*hue) + -0.04238, 2)

    #~~ Step 7.5: Find final organic carbon level (= Average of eq 1 and 2)
    Organic_Carbon = round(((SOC_SandySiltLoam+SOC_ClaySiltLoam)/2),2)

    #~~ Step 7.6: Find Bulk Density using clay and saturation inputs 
    BD = round((0.0129651*clayPercent) + (0.0030006*sat) + 0.4401499,2)

    #~~ Convert Organic Carbon from % to standard tC/ha format (total carbon per hectare)
    def tC_ha (depth, BD, SOC):
        depthC = depth*2.54
        tc_ha = round((SOC*0.01) * ((BD*(depthC/100)*10000)),2)
        return tc_ha

    print("Clay Percentage Level: ", clayPercent)
    print("Bulk Density (g/cm3): ", BD)
    print("Organic Carbon: ", Organic_Carbon, "%")
    print("Organic Carbon: ", tC_ha(depth, BD, Organic_Carbon), " tC/ha")

#~~ Step 8: Find Organic Matter Level
def Organic_Matter(root_image):
    img = cv2.imread(root_image, 1)

    #~~ Step 8.1: Convert image to HSV and extract values
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    avg_color_per_row = np.average(hsv, axis=0)
    avg_colors = np.average(avg_color_per_row, axis=0)
        
    hue = avg_colors[0]
    sat = avg_colors[1]
    val = avg_colors[2]
        
    #~~ Step 8.2: Calculate clay percentage 
    clayPercent = round((-0.0853*sat)+37.1,2)

    #~~ Step 8.3: Calculate Organic Carbon level for Fine sandy loam and Silt loam 
    SOC_SandySiltLoam = round((0.0772*hue) + 1.72, 2)
    if SOC_SandySiltLoam > 5.8:
        SOC_SandySiltLoam = 5.8

    #~~ Step 8.4: Calculate Organic Carbon level for Silt clay loam and Silt loam.
    try:
        SOC_ClaySiltLoam = round((0.05262*hue) + (0.11041*clayPercent) + -2.76983, 2)
    except:
        SOC_ClaySiltLoam = round((0.05902*hue) + -0.04238, 2)

    #~~ Step 8.5 Find final Organic Carbon level (= Average of eq 1 and 2)
    Organic_Carbon = round(((SOC_SandySiltLoam+SOC_ClaySiltLoam)/2),2)

    #~~ Step 8. Convert Organic Carbon level to Organic Matter level 
    def SOM(SOC):
        SOM = round(SOC * 1.72,2)
        if SOM > 10:
            SOM = 10
        return SOM

    print("Organic Matter Level: ", SOM(Organic_Carbon), "%")

#~~ Step 9: Run pretrained classification models
def copied_classifiers(root_image):
    image = cv2.imread(root_image)
    
    #~~ Step 9.1: Extract blue, red, green channels from image
    blue_channel = image[:,:,0]
    green_channel = image[:,:,1]
    red_channel = image[:,:,2]
    temp = ((np.median(green_channel)+np.median(blue_channel))+np.median(red_channel))
    temp = np.nanmean(temp)

    #~~ Step 9.2: Load Pretrained Models
    Pmodelclass = pickle.load(open('models\Pclassifier.pkl', 'rb'))
    pHmodelclass = pickle.load(open('models\pHclassifier.pkl', 'rb'))
    OMmodelclass = pickle.load(open('models\OMclassifier.pkl', 'rb'))
    ECmodelclass = pickle.load(open('models\ECclassifier.pkl', 'rb'))

    #~~ Step 9.3: Predict P,pH, SOM, and EC using pretrained models
    Presult = round(float(Pmodelclass.predict([[temp]])), 3)
    pHreuslt = round(float(pHmodelclass.predict([[temp]])), 3)
    OMresult = float(OMmodelclass.predict([[temp]]))
    ECresult = float(ECmodelclass.predict([[temp]]))
            
    result = {
    'Phosphorus':Presult, 
    'pH Level':pHreuslt, 
    'Organic Matter Level':OMresult, 
    'Electrical Conductivity':ECresult } 
    print(result)

#~~ Compile and Run all steps together
if soil_analysis == True:
    print("-------------------------------------")
    print("Beginning Soil Analysis...")
    #original_image = sys.argv[1]
    original_image = "Soil.png"
    ImgEnhancer(original_image)
    root_image = ("CLAHE_" + original_image)
    ColorPalette(root_image)
    ColorSpace(root_image)
    classical_ph(root_image)
    Modern_pH(root_image)
    texture(root_image)
    Organic_Carbon(root_image)
    Organic_Matter(root_image)
    copied_classifiers(root_image)
    print("Soil Analysis Completed")
    print("-------------------------------------")