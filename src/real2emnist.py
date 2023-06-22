#!/usr/bin/env python
# coding: utf-8

### TRANSFORM ORIGINAL DATA IN EMNIST FORMAT ###

# Setup
import pylab as pl
import numpy as np
import cv2
import os.path  # for save the image 
from PIL import Image, ImageEnhance, ImageFilter # Pillow: Nice tool for image processing
import pandas as pd
import matplotlib.pyplot as pyplot


# Load image location and names
general_path = '/home/ubuntu/nashome/PrivatLRZ/nashome/PrivatLRZ/'
#general_path = 'C:/Users/ru84fuj/Desktop/'
photonames = pd.read_excel(general_path + 'bunte_names.xlsx', index_col=None)
# Get labels from original data
#labels = photonames['text_inc']
# Get names of original images
file_names = photonames['orig_file_name']
# Get number of original images
file_numbers = photonames['image_number']
# Get names to save image
save_names = photonames['file_name']

# Take a look at the list
#print(f'Labels: \n{labels.head(10)} \n\nFile names: \n{file_names.head(10)}\n\nFile numbers: \n{file_numbers.head(10)}\n\nSave names: \n{save_names.head(10)}')
photonames.head()
print(f'Preprocessing {len(file_numbers)} images')

# Create function to increase brightness

def increase_brightness(img, value=20):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def contraAndSharp(img):
    #Contrast factor
    contrFactor = 5  # 1 = do nothing. We increase contrast

    # Sharpness factor
    SharpFactor = 5  # increase sharpness

    #data_contrast = Image.open(f"./PreProc1/{img}")

    enhancerContrast = ImageEnhance.Contrast(img)
    imageContrast = enhancerContrast.enhance(contrFactor)

    enhancerSharpness = ImageEnhance.Sharpness(imageContrast)
    imageSharp = enhancerSharpness.enhance(SharpFactor)

    final = imageSharp

    return final


for i in range(len(file_names)):
    # Select image
    image_path = '/home/ubuntu/nashome/PrivatLRZ/nashome/PrivatLRZ/bunte/' + file_names[i]
    
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 
    #img = cv2.imread(image_path) 
    
    # Invert image: Pay attention, only if the image is not already in emnist format!
    # This could potentially be checked with the pixel distribution
    
    # Consider if increasing brightness before as noise reduction method is better
    # Check also contrast functions
    
    # Convert to PIL format
    # For this it is important to change the color channel sequence.
    im_pil = Image.fromarray(img)

    # Get width and height and define top section that will remain after cropping
    # Think about an adaptable solution, e.g. only if there is no text, crop it
    w,h = im_pil.size
    top_remain = h/2.5
    
    # Crop and save with new name which correspond to labels
    cropped = im_pil.crop(( 0,0 , w  , top_remain ))
    
    # Increase contrast and sharpness
    cropped = contraAndSharp(cropped)
    
    #image brightness enhancer
    enhancer = ImageEnhance.Brightness(cropped)
    # Increase brightness
    factor = 3 #brightens the image
    cropped = enhancer.enhance(factor)
    
    array_img = np.array(cropped) 
    
    array_img[array_img>=255] = 255
    array_img[array_img<255] = 0
       
    # Invert
    output_img = array_img.copy()
    output_img[array_img == 255] = 0
    output_img[array_img != 255] = 255
    
    # Find max values
    #max_value = np.max(output_img)
    #min_value = np.min(output_img)
    #print(f'Max value: {max_value}\nMin value: {min_value}\nSet of values: {set(list(output_img.flatten()))}\n')
    
    
    pyplot.imsave(f'{general_path}/bunte_preprocessed/{save_names[i]}', output_img, cmap=pyplot.get_cmap('gray'))


