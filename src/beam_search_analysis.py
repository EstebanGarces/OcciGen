# -*- coding: utf-8 -*-

# SETUP

from transformers import GPT2Config, ViTConfig, VisionEncoderDecoderConfig, VisionEncoderDecoderModel, ViTFeatureExtractor, GPT2Tokenizer, PreTrainedTokenizerFast, AutoFeatureExtractor
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import requests 
from PIL import Image
from torchmetrics import CharErrorRate
import numpy as np

output_dir = '/home/ubuntu/nashome/PrivatLRZ/nashome/PrivatLRZ/swin_BERT_syn/'
model_dir = '/home/ubuntu/nashome/PrivatLRZ/nashome/PrivatLRZ/swin_BERT_syn/'
tokenizer_path = '/home/ubuntu/nashome/PrivatLRZ/nashome/PrivatLRZ/'
image_path = '/home/ubuntu/nashome/PrivatLRZ/nashome/PrivatLRZ/'

image_dataframe = pd.read_excel(image_path + 'test_40000.xlsx')
image_names = list(image_dataframe['file_name'])
image_raw_labels = list(image_dataframe['text_bert'])
#length_special_token = len('<|endoftext|>')
length_special_token = len('[SEP]')

image_labels = [raw_label[0:(len(raw_label) - length_special_token)] for raw_label in image_raw_labels]

# Own

# Load model
model_own = VisionEncoderDecoderModel.from_pretrained(model_dir)

# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

model_own.to(device)

feature_extractor_own = AutoFeatureExtractor.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k")
tokenizer_own = PreTrainedTokenizerFast(tokenizer_file= tokenizer_path + "byte-level-BPE.tokenizer.json")

# Define metric
metric = CharErrorRate()

# Randomized selection
num_examples = len(image_raw_labels)
import random
random.seed(42)
index_list = random.sample(range(len(image_names)), num_examples)
img_names = [image_names[i] for i in index_list]
img_labels = [image_labels[i] for i in index_list]


beam_width_list = [4, 10, 15, 30, 50]
#beam_width_list = [150]
output_names_list = ['beam_04_test_perf.xlsx', 'beam_10_test_perf.xlsx', 'beam_15_test_perf.xlsx', 'beam_30_test_perf.xlsx', 'beam_50_test_perf.xlsx']
#output_names_list = ['beam_150_test_perf.xlsx']

# Loop through different beam widths

for j in range(len(beam_width_list)):
    # Set beam search parameters
    # Empty predictions list
    labels_list = []
    predictions_list = []
    cer_list = []

    #model_own.config.num_beams = beam_width_list[j]
    beam_width = beam_width_list[j]
    print(f'------Calculating decoding with beam width of {beam_width}------')
    
    for i in range(num_examples):
        # Load image
        image = Image.open(image_path + 'test_preprocessed/' + img_names[i]).convert("RGB")
        # Load label
        label = img_labels[i]
        # Generate pixel values
        pixel_values_own = feature_extractor_own(image, return_tensors="pt").pixel_values.to(device) 
        # Generate ids predictions
        generated_ids_own = model_own.generate(pixel_values_own,
                              no_repeat_ngram_size=100,
                              num_beams = beam_width,
                              #do_sample = True, 
                              max_length = 2*100,            
                              #top_k = beam_width, 
                              #top_p = 0.85, 
                              num_return_sequences = 1)
        # Decode predictions
        generated_text_own = tokenizer_own.batch_decode(generated_ids_own, skip_special_tokens=True)[0]
        # Calculate CER
        cer_own = metric(preds = generated_text_own, target=label).item() # .item() to transform tensor to float
        # Round it
        cer_own = np.round(cer_own, 5)
        # Add labels, predictions and CER to lists
        labels_list.append(label)
        predictions_list.append(generated_text_own)
        cer_list.append(cer_own)
        print(f'(Beam width of {beam_width})')
        print(f'Test example number {i+1}:\nLabel: {label}\nPrediction: {generated_text_own}\nCER: {cer_own}\nCurrent mean CER: {np.mean(cer_list)}\n')
    

    mean_cer = np.mean(cer_list)
    # Round up to two decimal places
    mean_cer = np.round(mean_cer, 5)
    print(f'\nMean CER over {len(cer_list)} test examples: {mean_cer}\n')

    output_df = pd.DataFrame([labels_list, predictions_list, cer_list])
    output_df = output_df.transpose() #To Transpose and make each rows as columns
    output_df.columns=['Label','Prediction','CER'] #Rename the columns


    output_df.to_excel(output_dir + output_names_list[j])






