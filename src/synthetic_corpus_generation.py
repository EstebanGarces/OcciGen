# -*- coding: utf-8 -*-

import re
import pandas as pd
import random

# Creat function that take standard corpus words and convert them in the format "GRAPHICAL VARIANT -> LEMMA"
def get_word_pairs(filename, eos_token, seed):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
    words = re.findall(r'\b\w+\b', text)
    words = list(set([word.upper() for word in words]))
    # Random shuffle
    random.seed(seed)
    words_shuffled = random.sample(words, len(words))
    pairs = []
    for i in range(len(words_shuffled)):
        for j in range(i+1, len(words_shuffled)):
            pairs.append(words_shuffled[i] + '@' + words_shuffled[j] + eos_token)
    return pairs

# Specify file location
file_location = '/path/to/original_merged_corpora.txt'
custom_token = '[SEP]'
custom_token = ''
output_pairs = get_word_pairs(file_location, custom_token, 42)
train_pairs = output_pairs[:int(len(output_pairs)*0.8)]
test_pairs = output_pairs[int(len(output_pairs)*0.8):]

# Add pairs from training
labels_df = pd.read_excel('/path/to/blue_cards_labels.xlsx')
labels_df = labels_df[labels_df['split'] == 'train']
add_pairs = labels_df["label"] 
add_pairs = list(add_pairs)
add_pairs = set(add_pairs)
add_pairs = list(add_pairs)
random.seed(42)
add_pairs_shuffled = random.sample(add_pairs, len(add_pairs))
add_train = add_pairs[:int(len(add_pairs)*0.8)]
add_test = add_pairs[int(len(add_pairs)*0.8):]

# Add both lists
train_list = train_pairs + add_train
test_list = test_pairs + add_test

# Remove duplicates
train_list = set(train_list)
train_list = list(train_list)
test_list = set(test_list)
test_list = list(test_list)

# Shuffle the lists
train_list_shuffled = random.sample(train_list, len(train_list))
test_list_shuffled = random.sample(test_list, len(test_list))

# Join elements into a string
output_train_text = ' '.join(train_list_shuffled)
output_test_text = ' '.join(test_list_shuffled)

# Write the strings
def build_text_files(data_list, dest_path):
    f = open(dest_path, 'w')
    f.write(data_list)

# Apply function and fill text files    
build_text_files(output_train_text, '/path/to/synthetic_corpus.txt')
build_text_files(output_test_text, '/path/to/test_corpus.txt') # Optional

