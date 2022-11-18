import pandas as pd 
import numpy as np
import cv2
from skimage import data
from skimage.color import rgb2xyz, xyz2rgb
import PIL
import sys
from scipy import ndimage
from skimage import measure, color, io

def classical_ph(root_image):
    df=pd.read_csv('data\classic_pH_data.csv')    #import csv file
    g_frac=np.array(df.g_fraction)        #g_fraction values
    pH=np.array(df.pH)                    #pH values
    soil_data=[]                          #Array for g_fraction and pH values

    for i in range(len(pH)):
        sd=[g_frac[i], pH[i]]
        soil_data.append(sd)

    #Average RGB Calculation
    imgName = root_image 
    image = PIL.Image.open(imgName)   #import image
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
    g_frac_img=G_mean/(R_mean + G_mean + B_mean)  #g_fraction of image

    Data=[]             #Array for difference in g_fraction with corresponding pH
    Dif=[]              #Array for difference in g_fraction
    for i in soil_data:
        g_frac_dif = np.abs(i[0]-g_frac_img)
        Data.append([g_frac_dif, i[1]])
        Dif.append(g_frac_dif)

    for j in Data:
        if j[0]==min(Dif):
            print("pH value from classical rgb prediction is:", j[1])

#root_image = sys.argv[1]
root_image = "CLACHE_Soil.png"
classical_ph(root_image)