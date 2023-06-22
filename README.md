# Automatic transcription of handwritten Old Occitan language

# Old Occitan Dictionary

- Annotation of ~ 41k data points
- Training of a HTR model based on the transformer architecture
- Multiple experiments to obtain the best model (CER 0.005, Swin + BERT)
- Further experiments on an external data set

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- This file.
    ├── requirements.txt   <- The requirements.txt file including all dependencies.
    ├── .gitattributes     <- The extensions to be tracked with Git LFS.
    │
    ├── augmentation       <- Includes the synthetic corpora and the extended emnist data set.
    │
    ├── benchmarking       <- Includes benchmarking results with external OCR/HTR tools.
    │
    ├── data               <- Includes the merged corpora file, labels and original and preprocessed images for the DOM project and an external Occitan data set.
    │
    ├── model              <- Includes best-performing model with its config.json file.
    │
    ├── src                <- Contains the source code for training (model and tokenizer), inference, and data preparation.
    │
    └── tokenizer          <- Includes the byte-level BPE tokenizer and the corpus it is trained on.



--------


# Inference example in `python`:
```
import torch
import numpy as np
from PIL import Image
from transformers import VisionEncoderDecoderModel, PreTrainedTokenizerFast, AutoFeatureExtractor
from torchmetrics import CharErrorRate
import pandas as pd
import random

output_dir = '/path/to/output/directory/'
image_path = '/path/to/images/'
label_path = '/path/to/labels/'
tokenizer_path = '/path/to/tokenizer/'
model_path = '/path/to/model/'

image_dataframe = pd.read_excel(label_path + 'blue_cards_labels.xlsx')
image_dataframe = image_dataframe[image_dataframe['split'] == 'test']
image_names = image_dataframe['preproc_file_name'].tolist()
image_labels = image_dataframe['label'].tolist()

# Load model
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path + "byte-level-BPE.tokenizer.json")
model = VisionEncoderDecoderModel.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k")

# Define metric
metric = CharErrorRate()

# Initialize lists
labels_list = []
predictions_list = []
cer_list = []

num_examples = len(image_raw_labels)

# Run predictions and calculate CER
for i in range(num_examples):
    # Load image
    image = Image.open(image_path + 'test_preprocessed/' + image_names[i]).convert("RGB")
    # Load label
    label = image_labels[i]
    # Generate pixel values
    pixel_values = feature_extractor(image, return_tensors="pt").pixel_values.to(device)
    # Generate ids predictions
    generated_ids = model.generate(
        pixel_values,
        no_repeat_ngram_size = 100,
        num_beams = 1,
        max_length = 200,
        num_return_sequences = 1
    )
    # Decode predictions
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # Calculate CER
    cer = metric(preds=generated_text, target=label).item()
    # Round it
    cer = np.round(cer, 2)
    # Add labels, predictions, and CER to lists
    labels_list.append(label)
    predictions_list.append(generated_text)
    cer_list.append(cer)
    print(
        f'Test example number {i + 1}:\nLabel: {label}\nPrediction: {generated_text}\nCER: {cer}\nCurrent mean CER: {np.mean(cer_list)}\n'
    )

mean_cer = np.mean(cer_list)
mean_cer = np.round(mean_cer, 5)
print(f'\nMean CER over {len(cer_list)} test examples: {mean_cer}\n')

output_df = pd.DataFrame({'Label': labels_list, 'Prediction': predictions_list, 'CER': cer_list})
output_df.to_excel(output_dir + 'inference_results.xlsx', index=False)


```


