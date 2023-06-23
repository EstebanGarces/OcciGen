# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 22:19:25 2023

@author: ru84fuj
"""

import pandas as pd
import numpy as np

path = 'C:/Users/ru84fuj/Desktop/help_files/'

np.random.seed(45)  # Set random seed to 42

# Read the text file and extract words
with open(path + 'train_gpt2_mini.txt', 'r') as file:
    text = file.read()


forbidden_list = ['*', 'Ó', 'À', 'Á', 'Ṇ', 'Ï', 'È', 'É', "'", '-',
                  '(', '0', '?', '(']

words = set()
for word in text.split():
    # Check if word meets the criteria
    if (any(char.isdigit() or char.isalpha() or char == '@' or char == 'Ç' for char in word) and
            not any(forbidden_char in word for forbidden_char in forbidden_list)):
        words.add(word)


# Create a DataFrame from the extracted words
df = pd.DataFrame(words, columns=['Words'])

# Define the number of word pairs to sample
max_word_pairs = 180000

# Sample random words from the set
sampled_words = np.random.choice(list(words), size=max_word_pairs, replace=False)

# Create a DataFrame from the sampled words
df = pd.DataFrame(sampled_words, columns=['Words'])

# Function to print unique characters in sampled_words
def print_unique_characters(words):
    unique_chars = set()
    for word in words:
        unique_chars.update(set(word))
    print("Unique characters in sampled_words:", unique_chars)

# Print unique characters in sampled_words
print_unique_characters(sampled_words)


# Save DataFrame to Excel file
df.to_excel(path + 'names_for_data_aug.xlsx', index=False)