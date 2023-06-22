#-*- coding: utf-8 -*-

# SETUP

from transformers import GPT2Config, ViTConfig, VisionEncoderDecoderConfig, VisionEncoderDecoderModel, ViTFeatureExtractor, GPT2Tokenizer, PreTrainedTokenizerFast, TrOCRProcessor, AutoFeatureExtractor, DeiTFeatureExtractor, BeitFeatureExtractor
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import requests 
from PIL import Image
from torchmetrics import CharErrorRate
import numpy as np
import time

output_dir = '/home/ubuntu/nashome/PrivatLRZ/nashome/PrivatLRZ/swin_BERT_syn/' #Change the last part to the model directory
tokenizer_path = '/home/ubuntu/nashome/PrivatLRZ/nashome/PrivatLRZ/'
image_path = '/home/ubuntu/nashome/PrivatLRZ/nashome/PrivatLRZ/'

image_dataframe = pd.read_excel(image_path + 'test_40000.xlsx')
image_dataframe = pd.read_excel(image_path + 'bunte_names.xlsx')
image_names = list(image_dataframe['file_name'])
image_raw_labels = list(image_dataframe['text_bert'])
length_special_token = len('[SEP]')
image_labels = [raw_label[0:(len(raw_label) - length_special_token)] for raw_label in image_raw_labels]

# Load model
tokenizer_own = PreTrainedTokenizerFast(tokenizer_file= tokenizer_path + "byte-level-BPE.tokenizer.json")
model_own = VisionEncoderDecoderModel.from_pretrained(output_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model_own.to(device)

#feature_extractor_own = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
feature_extractor_own = AutoFeatureExtractor.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k")
#feature_extractor_own = DeiTFeatureExtractor.from_pretrained('facebook/deit-base-distilled-patch16-224')
#feature_extractor_own = BeitFeatureExtractor.from_pretrained('microsoft/beit-base-patch16-224-pt22k-ft22k')

# Define metric
metric = CharErrorRate()

# Make predictions

# Empty predictions list
labels_list = []
predictions_list = []
cer_list = []

num_examples = len(image_raw_labels)
import random
random.seed(42)
index_list = random.sample(range(len(image_names)), num_examples)
img_names = [image_names[i] for i in index_list]
img_labels = [image_labels[i] for i in index_list]

# Start time measure
start_time = time.time()

for i in range(len(image_names)*0+num_examples):
    # Load image
    #image = Image.open(image_path + 'test_preprocessed/' + img_names[i]).convert("RGB")
    image = Image.open(image_path + 'bunte_preprocessed/' + img_names[i]).convert("RGB")    
    # Load label
    label = img_labels[i]
    # Generate pixel values
    pixel_values_own = feature_extractor_own(image, return_tensors="pt").pixel_values.to(device) 
    # Generate ids predictions
    generated_ids_own = model_own.generate(pixel_values_own,
                              no_repeat_ngram_size=100,
                              num_beams = 1,
                              #do_sample = True, 
                              max_length = 2*100,            
                              #top_k = beam_width, 
                              #top_p = 0.85, 
                              num_return_sequences = 1)    
    # Decode predictions
    generated_text_own = tokenizer_own.batch_decode(generated_ids_own, skip_special_tokens=True)[0]
    #generated_text_own = predict[i]
    # Calculate CER
    cer_own = metric(preds = generated_text_own, target=label).item() # .item() to transform tensor to float
    # Round it
    cer_own = np.round(cer_own, 2)
    # Add labels, predictions and CER to lists
    labels_list.append(label)
    predictions_list.append(generated_text_own)
    cer_list.append(cer_own)
    print(f'Test example number {i+1}:\nLabel: {label}\nPrediction: {generated_text_own}\nCER: {cer_own}\nCurrent mean CER: {np.mean(cer_list)}\n')
    
# End time measure
end_time = time.time()

execution_time = np.round(end_time - start_time , 3)

print(f'Execution time: {execution_time} seconds')

execution_time_text = f'Execution time: {execution_time} seconds'

# Write the strings
def build_text_files(data_list, dest_path):
    f = open(dest_path, 'w')
    f.write(data_list)

build_text_files(execution_time_text, output_dir + 'bunte_inference_time.txt')

mean_cer = np.mean(cer_list)

# Round up to two decimal places
mean_cer = np.round(mean_cer, 2)
print(f'\nMean CER over {len(cer_list)} test examples: {mean_cer}\n')

output_df = pd.DataFrame([labels_list, predictions_list, cer_list])
output_df = output_df.transpose() #To Transpose and make each rows as columns
output_df.columns=['Label','Prediction','CER'] #Rename the columns
output_df.to_excel(output_dir + 'bunte_inference.xlsx')

