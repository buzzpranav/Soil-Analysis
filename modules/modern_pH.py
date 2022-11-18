import cv2
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import sys
import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Read image file as BGR
def Modern_pH(root_image):
    img = cv2.imread(root_image, 1)


    # Convert image from BGR to RGB and split channels
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    im = np.array(rgb_img) # get image as numpy array
    w,h,d = im.shape # get shape
    im.shape = (w*h, d) # change shape
    rgb_list = list(im.mean(axis=0)) # get average

    Red_val = rgb_list[0]
    Green_val = rgb_list[1]
    Blue_val = rgb_list[2]

    ds = pd.read_csv('data\modern_pH_data.csv')

    X = ds.iloc[:, :-1]
    X.loc[len(X)] = rgb_list
    X = X.values

    sc = StandardScaler()
    X = sc.fit_transform(X)

    model = joblib.load('models\modern_pH_model.pkl')
    pred = model.predict(X)[-1]
    return(Red_val, Blue_val, Green_val)
    print("The Modern pH reading is:", pred)

#root_image = sys.argv[1]
root_image = "CLACHE_Soil.png"
Modern_pH(root_image)