# Import packages and functions
import string
import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as pyplot
import os
from PIL import Image, ImageFilter

# Specify path to OcciGen

general_path = 'path/to/Occigen'
synthetic_corpus_path = general_path + 'augmentation/'
emnist_path = '/path/to/emnist/'

np.random.seed(45)  # Set random seed to 42

# Read the synthetic corpus file and extract words
with open(synthetic_corpus_path + 'synthetic_corpus.txt', 'r') as file:
    text = file.read()

# Specify characters that are not available in the extended EMNIST data set
forbidden_list = ['*', 'Ó', 'À', 'Á', 'Ṇ', 'Ï', 'È', 'É', "'", '-',
                  '(', '0', '?', '(']

words = set()
for word in text.split():
    # Check if word meets the criteria
    if (any(char.isdigit() or char.isalpha() or char == '@' or char == 'Ç' for char in word) and
            not any(forbidden_char in word for forbidden_char in forbidden_list)):
        words.add(word)

# Create a DataFrame from the extracted words
df = pd.DataFrame(words, columns=['Words'])

# Define how many synthetic images you want to generate
max_word_pairs = 180000

# Define number of snythetic images for each synthetic label
num_of_examples_per_label = 1

print(f'You will generate {max_word_pairs*num_of_examples_per_label} synthetic images')

# Sample random words from the set
sampled_words = np.random.choice(list(words), size=max_word_pairs, replace=False)

# Create a DataFrame from the sampled words
synthetic_labels = pd.DataFrame(sampled_words, columns=['Words'])

# Function to print unique characters in sampled_words
def print_unique_characters(words):
    unique_chars = set()
    for word in words:
        unique_chars.update(set(word))
    print("Unique characters in sampled_words:", unique_chars)

# Print unique characters in sampled_words
print_unique_characters(sampled_words)

# The synthetic labels will now be used for synthetic image generation

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
    
 
# Load extended EMNIST file
emnist_df = pd.read_csv(emnist_path + 'extended_emnist.csv', sep = ",")
print(set(emnist_df["label"]))

pixel_colnames = []
for i in range(784):
        pixel_colnames.append(f'pixel{i}')

print(emnist_df.max().max())

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


# Define output directory
output_dir = 'your/output/directory/'

# Generate synthetic images

synthetic_names = list(synthetic_labels['Words'])
os.mkdir(output_dir + 'synthetic_images')
start = time.time()
for i in range(len(synthetic_names)):
    pair = synthetic_names[i]
    for j in range(num_of_examples_per_label):
        pair_image = words2img(pair, pixels, labels, allow_multiline = True)
        # Save image
        final_path = output_dir + 'synthetic_images/'
        pyplot.imsave(f'{final_path}{pair}{j}.png', pair_image, cmap=pyplot.get_cmap('gray'))
    
print(f'The generation of the synthetic images took {time.time()-start} seconds')

# Create DataFrame for the synthetic images

file_name_list = []
label_list = []
label_eos_list = []
eos_token_bert = '[SEP]'

# Create DataFrame for synthetic images, ready for pre-training
for i in range(len(synthetic_names)):
    for j in range(num_of_examples_per_label):
        file_name_list.append(f'{synthetic_names[i]}{j}.png')
        label_list.append(f'{synthetic_names[i]}')
        label_eos_list.append(f'{synthetic_names[i]}{eos_token_bert}')        
        
output_df = pd.DataFrame(list(zip(file_name_list, label_list, label_eos_list)), columns = ['preproc_file_name', 'label', 'label_eos'])

output_df.to_excel(output_dir +"synthetic_images_and_names.xlsx", index = False)


