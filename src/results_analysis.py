# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt

# assume your data is stored in a pandas DataFrame called "df"

path_to_df = 'C:/Users/ru84fuj/Desktop/help_files/'

#df = pd.read_excel(path_to_df + 'test_results_0006.xlsx')
df_raw = pd.read_excel(path_to_df + 'swin_bert_vallari.xlsx')
df = df_raw[df_raw['Split'] == 'Test']


# 1. A list of characters that appear in the labels column
unique_chars = set(''.join(df['Label'].tolist()))

# 2. A summary where the dataframe is grouped by each of the characters, 
# and the average CER for the labels that contain that respective character.
grouped_df = pd.DataFrame(columns=['Character', 'Count', 'Mean CER', 'Median CER', 'Min CER', 'Max CER', 'Std CER'])

for char in unique_chars:
    regex = re.compile(re.escape(char))
    char_df = df[df['Label'].str.contains(regex)]
    count = char_df.shape[0]
    mean_cer = char_df['CER'].mean()
    mean_wcer = np.sum([char_df['Weighted_CER']/np.sum(char_df['Length'])])
    median_cer = char_df['CER'].median()
    min_cer = char_df['CER'].min()
    max_cer = char_df['CER'].max()
    std_cer = char_df['CER'].std()
    grouped_df = grouped_df.append({'Character': char, 'Count': count, 'Mean CER': mean_cer, 'Mean WCER': mean_wcer, 'Median CER': median_cer,
                                    'Min CER': min_cer, 'Max CER': max_cer, 'Std CER': std_cer},
                                   ignore_index=True)

# 3. Calculate separately the labels that only contain letters from A to Z and @.
letters_regex = re.compile(r'^[a-zA-Z@]+$')
letters_df = df[df['Label'].str.contains(letters_regex)]
letters_mean_cer = letters_df['CER'].mean()
letters_mean_wcer = np.sum([letters_df['Weighted_CER']/np.sum(letters_df['Length'])])
letters_median_cer = letters_df['CER'].median()
letters_min_cer = letters_df['CER'].min()
letters_max_cer = letters_df['CER'].max()
letters_std_cer = letters_df['CER'].std()
grouped_df = grouped_df.append({'Character': 'A-Z + @', 'Count': letters_df.shape[0], 'Mean CER': letters_mean_cer, 'Mean WCER': letters_mean_wcer,
                                'Median CER': letters_median_cer, 'Min CER': letters_min_cer, 'Max CER': letters_max_cer,
                                'Std CER': letters_std_cer}, ignore_index=True)

# 4. Sort the resulting dataframe by ascending Mean CER
grouped_df = grouped_df.sort_values(by='Mean WCER', ascending=True)

grouped_df.to_excel(path_to_df + "swin_bert_label_performance_by_char.xlsx")

from adjustText import adjust_text

# plot counts against CER
plt.figure(figsize=(10, 8))
sns.scatterplot(data=grouped_df, x='Count', y='Mean CER', hue='Character', s=100)

# add labels to the plot
texts = []
for i in range(len(grouped_df)):
    texts.append(plt.text(grouped_df['Count'][i] + 0.1, grouped_df['Mean CER'][i] + 0.001, grouped_df['Character'][i], fontsize=10))

plt.xlabel('Count')
plt.ylabel('Mean Weighted CER')
plt.title('Mean Weighted CER vs. Character frequency in labels')

# adjust label positions to avoid overlap
adjust_text(texts)

# create multi-column legend
handles, labels = plt.gca().get_legend_handles_labels()
num_columns = 2
num_labels = len(labels)
num_rows = (num_labels + num_columns - 1) // num_columns
legend = plt.legend(handles, labels, ncol=num_columns, bbox_to_anchor=(1.01, 1), borderaxespad=0)

# add footnote
plt.annotate('* "A-Z + @" excludes special characters such as diacritics, numbers and punctuation marks.', 
             xy=(0.5, -0.15), xycoords='axes fraction', fontsize=10, ha='center')


plt.show()

## Group by length

# create a new column for the length of the labels
df['Label Length'] = df['Length']

# group by label length and calculate mean CER
grouped_df = df.groupby('Label Length').agg({'CER': ['mean', 'mean', 'median', 'min', 'max', 'std'], 'Label': 'count'})
grouped_df.columns = ['Mean CER', 'Mean WCER', 'Median CER', 'Min CER', 'Max CER', 'Std CER', 'Count']
grouped_df = grouped_df.reset_index()

# create scatter plot of Mean CER vs label length
plt.figure(figsize=(10, 8))
sns.scatterplot(data=grouped_df, x='Label Length', y='Mean WCER', s=100)

# add labels to the plot
texts = []
for i in range(len(grouped_df)):
    texts.append(plt.text(grouped_df['Label Length'][i] + 0.1, grouped_df['Mean WCER'][i] + 0.001, grouped_df['Count'][i], fontsize=10))

plt.xlabel('Label Length')
plt.ylabel('Mean Weighted CER')
plt.title('Mean Weighted CER vs Label Length')

# adjust label positions to avoid overlap
adjust_text(texts)

plt.annotate('* The numbers represent the label frequency per length', 
             xy=(0.5, -0.15), xycoords='axes fraction', fontsize=10, ha='center')


# Check number of examples
print(f'Number of test examples: {np.sum(grouped_df["Count"])}')

plt.show()








