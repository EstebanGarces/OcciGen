# -*- coding: utf-8 -*-

# Import packages and functions
import pandas as pd
from torchmetrics import CharErrorRate
import numpy as np

from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='oc') # Occitan library

output_dir = '/your/output/directory/'

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

for i in range(num_examples):
    # Load image
    unit_path = image_path + 'blue_cards/' + img_names[i]
    # Load label
    label = img_labels[i]
    # Paddle OCR results
    result = ocr.ocr(unit_path, cls=True)
    pred = ""
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            pred += " " + line[1][0]
    generated_text = pred.upper()
    # Calculate CER
    cer = metric(preds = generated_text, target=label).item() # .item() to transform tensor to float
    # Round it up
    cer = np.round(cer, 6)
    # Add labels, predictions and CER to lists
    labels_list.append(label)
    predictions_list.append(generated_text)
    cer_list.append(cer)
    print(f'\nTest example number {i+1}:\nLabel: {label}\nPrediction: {generated_text}\nCER: {cer}\nCurrent mean CER: {np.mean(cer_list)}\n')
    
mean_cer = np.mean(cer_list)
# Round it up
mean_cer = np.round(mean_cer, 6)
print(f'\nMean CER over {len(cer_list)} test examples: {mean_cer}\n')

output_df = pd.DataFrame([labels_list, predictions_list, cer_list])
output_df = output_df.transpose() # To Transpose and make each rows as columns
output_df.columns=['Label','Prediction','CER'] # Rename the columns

output_df.to_excel(output_dir + 'paddle_ocr_test.xlsx', index = False)