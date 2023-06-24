# -*- coding: utf-8 -*-

# Import packages and functions
import cv2
import easyocr
import pandas as pd
from torchmetrics import CharErrorRate
import numpy as np

output_dir = '/your/output/directory/'

# EasyOCR reader
reader = easyocr.Reader(["oc"], gpu = False) # Occitan library

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
img_prep_names = [image_prep[i] for i in index_list]
img_labels = [image_raw_labels[i] for i in index_list]

def text_extraction(result):
    text_list= []
    for i in range(len(result)):
        text_list.append(result[i][1])
    return ' '.join(text_list).upper()
    
# Predictions and CER calculation
for i in range(num_examples):
    # Load image
    image = cv2.imread(image_path + 'test_preprocessed/' + img_prep_names[i])
    # Load label
    label = img_labels[i]
    # All result
    all_result = reader.readtext(image, paragraph = True)
    # Tesseract predictions
    generated_text = text_extraction(all_result)
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
mean_cer = np.round(mean_cer, 5)
print(f'\nMean CER over {len(cer_list)} test examples: {mean_cer}\n')

output_df = pd.DataFrame([labels_list, predictions_list, cer_list])
output_df = output_df.transpose() # To Transpose and make each rows as columns
output_df.columns=['Label','Prediction','CER'] # Rename the columns

output_df.to_excel(output_dir + 'easyocr_test_perf.xlsx')