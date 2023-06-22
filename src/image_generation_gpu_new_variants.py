# Import packages and functions
import string
import numpy as np
import emnist
import random
import time
import matplotlib.pyplot as pyplot
import os
import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import EMNIST
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.utils import make_grid
import matplotlib
import matplotlib.pyplot as plt
import pickle 
import requests
import zipfile
import os, shutil
from PIL import Image, ImageEnhance, ImageFilter

# Encode letters with numbers in a dictionary

abc = string.ascii_uppercase # A-Z
encode = {}
for i,j in enumerate(abc):
    encode[j] = i+10

encode['0'] = 0
encode['1'] = 1
encode['2'] = 2
encode['3'] = 3
encode['4'] = 4
encode['5'] = 5
encode['6'] = 6
encode['7'] = 7
encode['8'] = 8
encode['9'] = 9
encode['-'] = 36
encode['?'] = 37
encode['@'] = 38
encode['Ç'] = 39
encode[' '] = 40


# Define function for one lined generation

def single_line_img(enc_word, pixels, labels, enc = encode):    
# Gets an encoded list and provides a one-lined text
# e.g. [1,2,3,38,3,2,1] turns into "ABC@BCA"       
    length_singular = 28
    num_arrays = len(enc_word)
    
    # Initialization
    # Empty area of length from words
    full_area = np.zeros((length_singular,length_singular*num_arrays))
    
    # Fill first character
    if enc_word[0] == 0:
        sub_df = np.zeros([length_singular,length_singular])
    else:
        suby = np.where(labels == float(enc_word[0]))
        sub123 = random.choice(suby[0])
        sub_df = pixels[sub123]
    
    full_area[0:length_singular,0:length_singular] = sub_df

    # Fill rest of the characters with random shifts to the left
   
    for i in range(len(enc_word)-1):
        # First determine which character example will be added
        if enc_word[i+1] == 0:
            sub_df = np.zeros([length_singular,length_singular])
        else:
            suby = np.where(labels == float(enc_word[i+1]))
            #print(suby)
            sub123 = random.choice(suby[0])
            sub_df = pixels[sub123]
        
        # Add them with random shifts
        shift_left = random.choice([0,1,2,3,4,5,6,7,8])       
        full_area[:,length_singular*(i+1)-shift_left:(length_singular*(i+1)-shift_left+length_singular)]+=sub_df

    output = full_area
    # Normalization (0 or 255)  
    output[output>=255] = 255
    output[output<255] = 0
    
    return(output.reshape((length_singular,-1)))


# Define function to split input text into words and allow multiple line generation

def words2img(text, pixels, labels, enc = encode, allow_multiline = True):
    '''Receives a string and outputs a framed (multi-lined) image'''
    
    # Define size of characters
    length_singular = 28    
    
    # Randomly choose single or double line images
    allow_multiline = random.choice([0,1])
    
    # Read text
    word = text
    
    # Cover unknown characters
    letters1 = list(word)
    # Check if there are unknown characters and replace the with <UNK>
    newletters1 = []
    for letter in letters1:
        if letter in encode.keys():
            newletters1.append(letter)
        else:
            newletters1.append('<UNK>')
    
    # Generate encoded list, e.g. turns 'ABACA' into [1,2,1,3,1]
    letters = [encode[i] for i in newletters1]
    
    # Distinguish cases of single-lined text or multiple lined text        
    if allow_multiline == False:
        output_img = single_line_img(enc_word = letters, pixels = pixels, labels = labels, enc = encode)
    else:
        # Splits words in order to produce two line images
        # A mask of zeros is generated, e.g. "ACA@ABA"
        # letters = [1,3,1,38,1,2,1]
        # letters_first = [1,2,1,38,0,0,0]
        # letters_second = [0,0,0,0,1,2,1]
        # In the case of multiline, letters_first and letters_second are vertically stacked
        
        # Mode 1: Arrow only above
        # Mode 2: Arrow below
        arrow_pos = random.choice([1,2])
               
        word_list = word.split('@')
            
        if(arrow_pos == 1):
            len1 = len(word_list[0])   
            if len(word_list) == 2:
                len2 = len(word_list[1])
        elif(arrow_pos == 2):
            len1 = len(word_list[0])-1   
            if len(word_list) == 2:
                len2 = len(word_list[1])+1
            
        ids_left = letters[0:(len1+1)]
        ids_right = letters[(len1+1):len(letters)]
        zero_right = list([0 for letter in letters[(len1+1):len(letters)]])
        zero_left = list([0 for letter in letters[0:(len1+1)]])

        letters_first = ids_left
        for i in range(len(zero_right)):
            letters_first.append(zero_right[i])
                
        letters_second = zero_left
        for i in range(len(ids_right)):
            letters_second.append(ids_right[i])
                
        # Vertically stack single lined images    
        output_img = np.stack((single_line_img(enc_word = letters_first, pixels = pixels, labels = labels , enc = encode),
                           single_line_img(enc_word = letters_second, pixels = pixels, labels = labels, enc = encode)))
    
    # Reshape to a valid image format    
    output_img = output_img.reshape((length_singular*(allow_multiline+1),-1))
    
    # Random starting position
    start_top = np.int_(length_singular*random.choice([0,1,2,3])+1)
    start_left = np.int_(length_singular*random.choice([0,1,2,3])+1)
      
    # Min dimensions
    min_dim_vertical = np.int_(start_top+output_img.shape[0])
    min_dim_horizontal = np.int_(start_left+output_img.shape[1])
    
    # Calculate random dimensions of frame under min size constraints
    frame_dim_vertical = np.int_(min_dim_vertical + random.choice([0,1,2,3,4,5,6])*length_singular)
    frame_dim_horizontal = np.int_(min_dim_horizontal + random.choice([0,1,2,3,4,5,6])*length_singular)

    # Initialize frame
    framed_img = np.zeros([frame_dim_vertical,frame_dim_horizontal])
    
    # Fill frame with image
    framed_img[start_top:(start_top+output_img.shape[0]), start_left:(start_left+output_img.shape[1])] = output_img
    
    # Transform to PIL to perform image modifications easily
    pil_img = Image.fromarray(framed_img)
    
    # Change thickness of text at random levels
    kernel = random.choice([1,3])
    pil_img = pil_img.filter(ImageFilter.MaxFilter(kernel))
    
    # Rotate text with random angles
    angle = random.choice([-3, -2, -1, 0, 1, 2, 3])
    rotated_img = pil_img.rotate(angle, expand = True)
    
    # Resize image
    #basewidth = random.choice([1000])
    #wpercent = (basewidth/float(rotated_img.size[0]))
    #hsize = int((float(rotated_img.size[1])*float(wpercent)))
    #resized_img = rotated_img.resize((basewidth,hsize), Image.Resampling.LANCZOS)
           
    # From PIL to Array
    array_img = np.array(rotated_img) 
    
    return(array_img)
    
    

# Load dataframes

output_dir = '/home/ubuntu/nashome/PrivatLRZ/nashome/PrivatLRZ/'
emnist_dir = '/home/ubuntu/nashome/PrivatLRZ/nashome/PrivatLRZ/'

train_digits_uppercase = pd.read_csv(emnist_dir + 'train_digits_uppercase.csv', sep = ",")
test_digits_uppercase = pd.read_csv(emnist_dir + 'test_digits_uppercase.csv', sep = ",")
special_char = pd.read_csv(emnist_dir + 'emnist_special_char.csv', sep = ";")

#tuple_shape_train = (len(train_digits_uppercase), 785)
#train_digits_uppercase.shape = tuple_shape_train


# Take a look at data structure
print(train_digits_uppercase.head())
print(test_digits_uppercase.head())
print(special_char.head())

# Select only classes lower than 36 in train and test
#train_digits_uppercase = train_digits_uppercase[train_digits_uppercase["label"] <= 35]
#test_digits_uppercase = test_digits_uppercase[test_digits_uppercase["label"] <= 35]

#train_digits_uppercase = train_digits_uppercase.astype(float)
#test_digits_uppercase = test_digits_uppercase.astype(float)
#special_char = special_char.astype(float)

from sklearn.preprocessing import MinMaxScaler

#scaler = MinMaxScaler()

pixel_colnames = []
for i in range(784):
        pixel_colnames.append(f'pixel{i}')

#train_digits_uppercase[pixel_colnames] = scaler.fit_transform(train_digits_uppercase[pixel_colnames])*255
#test_digits_uppercase[pixel_colnames] = scaler.fit_transform(test_digits_uppercase[pixel_colnames])*255
#special_char[pixel_colnames] = scaler.fit_transform(special_char[pixel_colnames])*255

print(train_digits_uppercase.max().max())
print(test_digits_uppercase.max().max())
print(special_char.max().max())


# Bring train, test and special characters df together
emnist_df = pd.concat([train_digits_uppercase, test_digits_uppercase])
emnist_df = pd.concat([emnist_df, special_char])
#emnist_df = emnist_df.astype(int)

label_colnames = ['label']
labels = emnist_df[label_colnames].to_numpy()

labels_shape = emnist_df.shape[0]
labels.shape = (labels_shape,)

pixels = emnist_df[pixel_colnames].to_numpy()

# Binarization
threshold = 127
pixels[pixels>=threshold] = 255
pixels[pixels<threshold] = 0

pixels.shape = (labels_shape, 28, 28)

from sklearn.model_selection import train_test_split
pixels_train, pixels_test, labels_train, labels_test = train_test_split(pixels, labels,
                                                    stratify=labels, 
                                                    test_size=0.5)

# Synthetic training data set

path = '/home/ubuntu/ocr-project/data_augmentation/'

num_examples = 1

#synthetic_train = pd.read_excel(path + 'new_variant_pairs_with_labels.xlsx')
synthetic_train = pd.read_excel('/home/ubuntu/nashome/PrivatLRZ/nashome/PrivatLRZ/image_names_aug.xlsx')
synthetic_train = list(synthetic_train['text_inc'])
os.mkdir(output_dir + 'synthetic_train')
start = time.time()
for i in range(len(synthetic_train)):
    # print(i)
    # number = random.choice(synthetic_train)
    pair = synthetic_train[i]
    #print(number)
    for j in range(num_examples):
        pair_image = words2img(pair, pixels_train, labels_train, allow_multiline = True)
        # Save image
        final_path = output_dir + 'synthetic_train/'
        pyplot.imsave(f'{final_path}{pair}{j}.png', pair_image, cmap=pyplot.get_cmap('gray'))
    
print(time.time()-start)
