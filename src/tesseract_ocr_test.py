# -*- coding: utf-8 -*-

# Import packages and functions
from PIL import Image
import pytesseract
import pandas as pd
from torchmetrics import CharErrorRate
import numpy as np

output_dir = '/your/output/directory/'

# Load Pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

# Create image path for test set
image_path = '/path/to/images/'
label_path = '/path/to/labels/'

# Read labels and image names
image_dataframe = pd.read_excel(label_path + 'blue_cards_labels.xlsx')
image_dataframe = image_dataframe[image_dataframe["split"] == "test"]
image_names = list(image_dataframe['orig_file_name'])
image_prep = list(image_dataframe['preproc_file_name'])
image_raw_labels = list(image_dataframe['label'])

# Define metric
metric = CharErrorRate()

# Empty predictions list
labels_list = []
predictions_list = []
cer_list = []

num_examples = len(image_raw_labels)
import random
random.seed(42)
index_list = random.sample(range(len(image_names)), num_examples)
img_names = [image_names[i] for i in index_list]
img_labels = [image_raw_labels[i] for i in index_list]

# Make predictions
for i in range(num_examples):
    # Load image
    image = Image.open(image_path + 'blue_cards/' + img_names[i]).convert("RGB")
    # Load label
    label = img_labels[i]
    # Tesseract predictions
    generated_text = pytesseract.image_to_string(image, lang = 'oci') # Occitan library
    # Calculate CER
    cer = metric(preds = generated_text, target=label).item() # .item() to transform tensor to float
    # Round it up
    cer = np.round(cer, 5)
    # Add labels, predictions and CER to lists
    labels_list.append(label)
    predictions_list.append(generated_text)
    cer_list.append(cer)
    print(f'Test example number {i+1}:\nLabel: {label}\nPrediction: {generated_text}\nCER: {cer}\nCurrent mean CER: {np.mean(cer_list)}\n')
    
mean_cer = np.mean(cer_list)
# Round it up
mean_cer = np.round(mean_cer, 2)
print(f'\nMean CER over {len(cer_list)} test examples: {mean_cer}\n')

output_df = pd.DataFrame([labels_list, predictions_list, cer_list])
output_df = output_df.transpose() # To Transpose and make each rows as columns
output_df.columns=['Label','Prediction','CER'] # Rename the columns

output_df.to_excel(output_dir + 'pytesseract_test.xlsx')