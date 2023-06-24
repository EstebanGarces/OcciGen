# -*- coding: utf-8 -*-

# SETUP

from transformers import VisionEncoderDecoderModel, PreTrainedTokenizerFast, TrOCRProcessor
import pandas as pd
import torch
from PIL import Image
from torchmetrics import CharErrorRate
import numpy as np

# Define output repository, model_path and tokenizer_path 
output_dir = '/your/output/directory/'
model_path = '/path/to/model'
tokenizer_path = '/path/to/tokenizer/'

# Create image path for test set
image_path = '/path/to/images/'
label_path = '/path/to/labels/'

# Read labels and image names
image_dataframe = pd.read_excel(label_path + 'blue_cards_labels.xlsx')
image_dataframe = image_dataframe[image_dataframe["split"] == "test"]
image_names = list(image_dataframe['orig_file_name'])
image_prep = list(image_dataframe['preproc_file_name'])
image_raw_labels = list(image_dataframe['label'])

# Load model, use own tokenizer and fine-tuned TrOCR with processor
tokenizer = PreTrainedTokenizerFast(tokenizer_file= tokenizer_path + "byte-level-BPE.tokenizer.json")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
processor.tokenizer = tokenizer
model = VisionEncoderDecoderModel.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

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

# Predictions and CER calculation
for i in range(num_examples):
    # Load image
    image = Image.open(image_path + 'test_preprocessed/' + img_names[i]).convert("RGB")
    # Load label
    label = img_labels[i]
    # Generate pixel values
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device) 
    # Generate ids predictions
    generated_ids = model.generate(pixel_values,
                              no_repeat_ngram_size=100,
                              num_beams = 1,
                              max_length = 200,            
                              num_return_sequences = 1)
    # Decode predictions
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
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
output_df = output_df.transpose() #To Transpose and make each rows as columns
output_df.columns=['Label','Prediction','CER'] #Rename the columns

# Export results
output_df.to_excel(output_dir + 'trocr_test.xlsx')



