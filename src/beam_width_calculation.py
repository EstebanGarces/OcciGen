# -*- coding: utf-8 -*-

# SETUP

from transformers import VisionEncoderDecoderModel, PreTrainedTokenizerFast, AutoFeatureExtractor
import pandas as pd
import torch
from PIL import Image
from torchmetrics import CharErrorRate
import numpy as np

# Specify paths
output_dir = '/your/output/path/'
model_path = '/path/to/model/'
tokenizer_path = '/path/to/tokenizer/'
image_path = '/path/to/images/'
label_path = '/path/to/labels/'

# Load image names and labels
image_dataframe = pd.read_excel(label_path + 'blue_cards_labels.xlsx')
image_dataframe = image_dataframe[image_dataframe["split"] == "text"]
image_names = list(image_dataframe['preproc_file_name'])
image_raw_labels = list(image_dataframe['label_eos'])
length_special_token = len('[SEP]')
image_labels = [raw_label[0:(len(raw_label) - length_special_token)] for raw_label in image_raw_labels]

# Load model, feature extractor and tokenizer
model= VisionEncoderDecoderModel.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k")
tokenizer = PreTrainedTokenizerFast(tokenizer_file= tokenizer_path + "byte-level-BPE.tokenizer.json")

# Define metric
metric = CharErrorRate()

# Define number of examples
num_examples = len(image_raw_labels)

# Define beam widths to be analyzed and names of output files
beam_width_list = [1, 4, 10, 15, 30, 50]
output_names_list = ['beam_01_test_perf.xlsx', 'beam_04_test_perf.xlsx', 'beam_10_test_perf.xlsx', 'beam_15_test_perf.xlsx', 'beam_30_test_perf.xlsx', 'beam_50_test_perf.xlsx']

# Loop through different beam widths

for j in range(len(beam_width_list)):

    # Empty output lists
    labels_list = []
    predictions_list = []
    cer_list = []

    beam_width = beam_width_list[j]
    print(f'------Calculating decoding with beam width of {beam_width}------')
    
    for i in range(num_examples):
        # Load image
        image = Image.open(image_path + 'test_preprocessed/' + image_names[i]).convert("RGB")
        # Load label
        label = image_labels[i]
        # Generate pixel values
        pixel_values = feature_extractor(image, return_tensors="pt").pixel_values.to(device) 
        # Generate ids predictions
        generated_ids = model.generate(pixel_values,
                              no_repeat_ngram_size=100,
                              num_beams = beam_width,
                              max_length = 200,            
                              num_return_sequences = 1)
        # Decode predictions
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # Calculate CER
        cer = metric(preds = generated_text, target=label).item() # .item() to transform tensor to float
        # Round it up
        cer = np.round(cer, 5)
        # Add labels, predictions and CER to lists
        labels_list.append(label)
        predictions_list.append(generated_text)
        cer_list.append(cer)
        print(f'(Beam width of {beam_width})')
        print(f'Test example number {i+1}:\nLabel: {label}\nPrediction: {generated_text}\nCER: {cer}\nCurrent mean CER: {np.mean(cer_list)}\n')
    

    mean_cer = np.mean(cer_list)
    # Round it up
    mean_cer = np.round(mean_cer, 5)
    print(f'\nMean CER over {len(cer_list)} test examples: {mean_cer}\n')

    output_df = pd.DataFrame([labels_list, predictions_list, cer_list])
    output_df = output_df.transpose() #T o Transpose and make each rows as columns
    output_df.columns=['Label','Prediction','CER'] # Rename the columns

    # Output results files
    output_df.to_excel(output_dir + output_names_list[j])






