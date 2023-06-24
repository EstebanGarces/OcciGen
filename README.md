# Automatic transcription of handwritten Old Occitan language

We propose an innovative HTR approach that leverages the transformer architecture for recognizing handwritten Old Occitan, a low-resource language. We develop and rely on elaborate data augmentation techniques for both text and image data.

Our model combines a custom-trained `Swin` image encoder with a `BERT` text decoder, which we pre-train using a large-scale augmented synthetic data set and fine-tune on the small human-labeled data set.

Experimental results reveal that our approach surpasses the performance of current state-of-the-art models for Old Occitan HTR, including open-source transformer-based models such as a fine-tuned `TrOCR` and commercial applications like `Google Cloud Vision`.

## Main steps

- Annotation of ~ 41k data points (80% train / 10% validation / 10% test)
- Pre-processing of images and text to enhance data augmentation:
    - Synthetic image generation through an extended `EMNIST dataset`
    - Synthetic corpus generation through merged Old Occitan corpora
- Byte-level BPE Tokenization on a synthetic Old Occitan corpus
- `VisionEncoderDecoder` model pre-training with synthetic images and fine-tuning with real (rotated and dilated) images
- Model selection from 24 experimental setups covering four vision encoders (`BEiT`, `DeiT`, `ViT` and `Swin`) and two language decoders (`GPT-2` and `BERT`)
- `Swin + BERT` is the best-performing model with a test performance of weighted CER 0.005 and 96.5% correctly predicted labels.
- Benchmarking against external open-source and commercial tools: `EasyOCR`, `PaddleOCR`, `Tesseract OCR`, `Google Cloud Vision`, and a fine-tuned `TrOCR`
  - Our approach achieves SOTA results in the Old Occitan data set
- Further experiments on an external data set exhibit a good performance (after post-processing) of weighted CER 0.011, and 92.1% correctly predicted labels 

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- This file.
    ├── requirements.txt   <- The requirements.txt file
    ├── .gitattributes     <- The extensions to be tracked with Git LFS.
    │
    ├── augmentation       <- Includes the synthetic corpora and the extended EMNIST data set.
    │
    ├── benchmarking       <- Includes benchmarking results with external OCR/HTR tools as well as benchmarking on an external Old Occitan data set.
    │
    ├── data               <- Includes the merged corpora file, labels, and original and preprocessed images for the DOM project and an external Occitan data set.
    │
    ├── error_analysis     <- Includes file with the test set performance of our final model. Used for visualizations and benchmarking.
    │    
    ├── model              <- Includes best-performing model (Swin + BERT) with its config.json file.
    │
    ├── src                <- Contains the source code for training (model and tokenizer), inference, data preparation, and benchmarking evaluation.
    │
    └── tokenizer          <- Includes the byte-level BPE tokenizer and the corpus it is trained on.



--------


# Example: Swin + BERT inference and CER calculation in `python`:
```
import os
import torch
from torch.utils.data import Dataset
from transformers import VisionEncoderDecoderModel, PreTrainedTokenizerFast, AutoFeatureExtractor
from PIL import Image
import pandas as pd
from torchmetrics import CharErrorRate
import numpy as np


# Specify the path to the OcciGen repository
repository_path = '/path/to/OcciGen/'

# Construct the paths relative to the repository
tokenizer_path = os.path.join(repository_path, 'tokenizer/')
model_path = os.path.join(repository_path, 'model/')
image_path = os.path.join(repository_path, 'data/dom_project/test_preprocessed/')
label_path = os.path.join(repository_path, 'data/dom_project/blue_cards_labels.xlsx')

image_dataframe = pd.read_excel(label_path)
image_dataframe = image_dataframe[image_dataframe["split"] == "test"]
image_names = list(image_dataframe['preproc_file_name'])
image_labels = list(image_dataframe['label'])

# Load model
tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path + "byte-level-BPE.tokenizer.json")
feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k")
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

# Define handling of data samples
class CustomDataset(Dataset):
    def __init__(self, image_names, image_path, feature_extractor, transform=None):
        self.image_names = image_names
        self.image_path = image_path
        self.feature_extractor = feature_extractor
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image = Image.open(self.image_path + image_name)
        
        # Convert image to RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        pixel_values = self.feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
        return pixel_values
    

dataset = CustomDataset(image_names, image_path, feature_extractor, transform=None)

# Inference and CER calculation
for i in range(len(image_labels)):
    pixel_values = dataset[i].to(device)  # Move the image tensor to the device
    try:
        # Generate ids predictions
        generated_ids = model.generate(
            pixel_values,
            no_repeat_ngram_size=100,
            max_length=200,
            num_beams=1,
            num_return_sequences=1
        )

        # Decode predictions
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # Calculate CER for each prediction in the batch
        label = image_labels[i]
        generated_text_i = generated_text[0]  # Since num_return_sequences=1, take the first generated sequence
        cer = metric(preds=generated_text_i, target=label).item()
        cer = np.round(cer, 5)

        labels_list.append(label)
        predictions_list.append(generated_text_i)
        cer_list.append(cer)

        print(f'Test example number {len(cer_list)}:\nLabel: {label}\nPrediction: {generated_text_i}\nCER: {cer}\nCurrent mean CER: {np.mean(cer_list)}\n')

    except TypeError as e:
        typekey = (pixel_values.shape, pixel_values.dtype)
        raise TypeError(f"Cannot handle this data type: {str(typekey[0])}, {str(typekey[1])}") from e

mean_cer = np.mean(cer_list)
mean_cer = np.round(mean_cer, 5)
print(f'\nMean CER over {len(cer_list)} test examples: {mean_cer}\n')

# Specify output directory
output_dir = '/your/output/directory/'

# Summarize and export inference results
output_df = pd.DataFrame({'Label': labels_list, 'Prediction': predictions_list, 'CER': cer_list})
output_df.to_excel(output_dir + 'inference_results.xlsx')


```


