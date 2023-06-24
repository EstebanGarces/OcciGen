# -*- coding: utf-8 -*-

from typing import Sequence
from google.cloud import vision
import pandas as pd
from torchmetrics import CharErrorRate
import numpy as np

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

# Define functions

import io

def detect_handwritten_ocr(path):
    """Detects handwritten characters in a local image.

    Args:
    path: The path to the local file.
    """
    from google.cloud import vision_v1p3beta1 as vision
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    # Language hint codes for handwritten OCR:
    # en-t-i0-handwrit, mul-Latn-t-i0-handwrit
    # Note: Use only one language hint code per request for handwritten OCR.
    image_context = vision.ImageContext(
        language_hints=['en-t-i0-handwrit'])

    response = client.document_text_detection(image=image,
                                              image_context=image_context)

    print('Full Text: {}'.format(response.full_text_annotation.text))
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            print('\nBlock confidence: {}\n'.format(block.confidence))

            for paragraph in block.paragraphs:
                print('Paragraph confidence: {}'.format(
                    paragraph.confidence))

                for word in paragraph.words:
                    word_text = ''.join([
                        symbol.text for symbol in word.symbols
                    ])
                    print('Word text: {} (confidence: {})'.format(
                        word_text, word.confidence))

                    for symbol in word.symbols:
                        print('\tSymbol: {} (confidence: {})'.format(
                            symbol.text, symbol.confidence))

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
        
def detect_handwritten_ocr_full(path):
    """Detects handwritten characters in a local image.

    Args:
    path: The path to the local file.
    """
    from google.cloud import vision_v1p3beta1 as vision
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    # Language hint codes for handwritten OCR:
    # en-t-i0-handwrit, mul-Latn-t-i0-handwrit
    # Note: Use only one language hint code per request for handwritten OCR.
    image_context = vision.ImageContext(
        language_hints=['en-t-i0-handwrit'])

    response = client.document_text_detection(image=image,
                                              image_context=image_context)

    #print('Full Text: {}'.format(response.full_text_annotation.text))
    output_text = response.full_text_annotation.text
    
    
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))
    return output_text
        

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
    unit_path = image_path + img_names[i]
    # Load label
    label = img_labels[i]
    # Create request and generate predictions
    # Google cloud vision predictions
    generated_text = detect_handwritten_ocr_full(unit_path)
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

output_dir = '/your/output/directory/'

output_df.to_excel(output_dir + 'google_cloud_vision_test.xlsx')

