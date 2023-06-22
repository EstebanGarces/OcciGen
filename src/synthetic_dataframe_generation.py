# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 22:43:50 2023

@author: ru84fuj
"""

import pandas as pd

#path = 'C:/Users/ru84fuj/Desktop/ocr-project/data_augmentation/'
path = '/home/ubuntu/nashome/PrivatLRZ/nashome/PrivatLRZ/'

# Load labels (training)

train_df = pd.read_excel(path + 'image_names_aug.xlsx')

num_examples = 1

file_name_list = []
text_inc_list = []
text_list = []
text_bert_list = []
#image_number_list = []

eos_token = '<|endoftext|>'
eos_token_bert = '[SEP]'

labels_list = train_df['text_inc']

#labels_unique = list(train_df['text_inc'])
#image_number_unique = list(train_df['image_number'])

for i in range(len(labels_list)):
    for j in range(num_examples):
        file_name_list.append(f'{labels_list[i]}{j}.png')
        text_inc_list.append(f'{labels_list[i]}')
        text_list.append(f'{labels_list[i]}{eos_token}')
        text_bert_list.append(f'{labels_list[i]}{eos_token_bert}')        
        
output_df = pd.DataFrame(list(zip(file_name_list, text_inc_list, text_list, text_bert_list)), columns = ['file_name', 'text_inc', 'text', 'text_bert'])

output_df.to_excel(path+"data_frame_image_names_aug.xlsx", index = False)
