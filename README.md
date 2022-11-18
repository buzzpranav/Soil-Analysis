# SOIL ANALYSIS
Author: Pranav S Narayanan
Copyright (c) [2022] [\[Pranav S Narayanan\]](https://buzzpranav.github.io)
License: MIT License

  

## Overview
This program uses a picture of soil to quickly and accurately find soil composition and relevant information such as pH Value, Soil Roughness, Clay Percentage, Bulk Density, Organic Carbon Levels, Organic Matter Levels, Phosphorus quantities, Electrical Conductivity of the soil. It then uses this information to create a detaile yet easy to understand report which farmers can use to plant the right crops and adequete fertilizers. The program can additionally use it's own report to suggest crops, fertilizers, planting times, and more.

  

## Introduction
All living things depend on soil. Healthy soil is essential for healthy plant growth, human nutrition, and water filtration. Healthy soil also prevent drought, floods, fires and even regulates the Earth's climate. Inarguably, the most important use of soil, is it’s use in agriculture.
From the first human civilizations to the modern day, agriculture has been the root of the economy, and soil is the root of agriculture. Anyone can throw a few seeds into the ground and get some crops, but to be economically viable and truly helpful for humanity, we must use high quality soil with proper concentrations of nutrients, along with fertilizers, and proper agriculture techniques to get the maximum crop yield. Having been in agriculture for thousands of years, farmers have definitely mastered the best techniques of it.

  

## The Problem
WIth the techniques of agriculture mastered, the only thing standing between humans and the solution to world hunger, climate change, and crop surplus is knowing about how much nutrients, fertilizers, etc. is already in the soil, and how much more is needed for the soil. Fertilizers, artificial yield boosters, and nutrients, when given in an overdosed will be harmful for plants, as well everything that consumes them. So how do we find this balance between underusing and overdosing fertilizers? The answer lies in soil analysis.
Imagine you just take a handful of dirt, send it to a lab, and immediately find out the nutrient composition. With access to this kind of technology, we could easily maximize crop yield.
There’s just 1 problem: No farmer would want to pay a couple hundred dollars, go through the hassle of packing and sending the soil to a lab, waiting a couple of days for the analysis, only to be thrown a massive report which makes no sense to them.

  

## Our Solution
We believe that our app and the machine learning algorithm at it’s core could be the next solution. Take a picture of your soil. Wait 60 seconds, and presto, you get a fully detailed, easy to understand report on soil pH values, Organic Matter levels, Salinity Percentage, Phosphorus and nutrient amounts using state of the art deep learning computer vision algorithms. Combine this informtion with the ages of experience and knowledge from farmers and we get the most powerful and viable agriculture economy the world has ever seen. 
This may seem very complex, or even impossible, but its not. Artificial Intelligence and Deep Learning, at its heart, is just finding patterns, mixed with some statistics and probability. All you need to do is take a picture and leave the rest to our app.
World Hunger. Climate Change. Poverty. Food Shortage. All fixed, 1 picture at a time.

  

## Methodology Overview
After taking a picture, the image will get preprocessed, where the ROI (Region of Interest) is snipped out. The snipped part will be deblurred and the lighting, fixed. The pH value can quickly be found by a quick calculation of the color of the soil. After that, our app uses TensorFlow, Sklearn, NumPy, OpenCV, and other python libraries to form around 10 separate AI regression models. CNN (Convolutional Neural Networks), RFR (Random Forest Regression), DTR (Decision Tree Regression) and SVR (Support Vector Regression) regression models are used to find organic matter, salinity, phospous, and pH amounts. A CNN classification algorithm is then fed the values of all the regression models to output an accurate value for all the predictions to a clean, readable, GUI. Furthermore, using it’s own data, the models can give suggestions on fertilizers to use or crops to sow. All this, in less than 60 seconds.




## Methodology
