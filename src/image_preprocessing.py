# coding: utf-8

### TRANSFORM ORIGINAL DATA IN EMNIST FORMAT ###

# Setup
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import pandas as pd
import matplotlib.pyplot as pyplot

# Define path to OcciGen
general_path = '/path/to/OcciGen/'

# Load image location and names
photonames = pd.read_excel(general_path + 'data/dom_project/blue_cards_labels.xlsx', index_col=None)
# Get names of original images
file_names = photonames['orig_file_name']
# Get number of original images
file_numbers = photonames['image_number']
# Get names to save image
save_names = photonames['preproc_file_name']

# Explore images
photonames.head()
print(f'Preprocessing {len(file_numbers)} images')

# Define functions for image enhancement: Brightness, Contrast and Sharpness

def increase_brightness(img, value=20):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    
    # Return enhanced image
    return img

def contraAndSharp(img):
    #Contrast factor
    contrFactor = 5  # 1 = do nothing. We increase contrast

    # Sharpness factor
    SharpFactor = 5  # 1 = do nothing. We increase sharpness

    # Apply factors
    enhancerContrast = ImageEnhance.Contrast(img)
    imageContrast = enhancerContrast.enhance(contrFactor)

    enhancerSharpness = ImageEnhance.Sharpness(imageContrast)
    imageSharp = enhancerSharpness.enhance(SharpFactor)

    # Return enhanced image
    return imageSharp

# Define output directory
output_dir = '/your/output/directory/'

# Perform pre-processing
for i in range(len(file_names)*0+5):
    # Select image
    image_path = general_path + 'data/dom_project/blue_cards/' + file_names[i]
    
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 
    
    # Convert to PIL format
    im_pil = Image.fromarray(img)

    # Crop and keep upper 40%
    w,h = im_pil.size
    top_remain = h/2.5 
    cropped = im_pil.crop(( 0,0 , w  , top_remain ))
    
    # Increase contrast and sharpness
    cropped = contraAndSharp(cropped)
    
    # Enhance brightness
    enhancer = ImageEnhance.Brightness(cropped)
    factor = 3 
    cropped = enhancer.enhance(factor)
    
    # Convert to array
    array_img = np.array(cropped) 

    # Binarize in EMNIST format    
    array_img[array_img>=255] = 255
    array_img[array_img<255] = 0
       
    output_img = array_img.copy()
    output_img[array_img == 255] = 0
    output_img[array_img != 255] = 255
    
    # Save results
    pyplot.imsave(f'{output_dir}{save_names[i]}', output_img, cmap=pyplot.get_cmap('gray'))


