# Setup

from transformers import BertConfig, VisionEncoderDecoderModel, BertTokenizer, VisionEncoderDecoderConfig, PreTrainedTokenizerFast, SwinConfig, SwinModel, AutoFeatureExtractor
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
# import wandb

# Specify paths to labels
path_labels = '/path/to/labels/'
tokenizer_path = '/path/to/tokenizer/'
model_path= '/path/to/model/'
output_dir = '/path/to/output/'

# Load labels (training and validation)
train_df = pd.read_excel(path_labels + 'blue_cards_labels.xlsx', header=0)
train_df = train_df[train_df["split"] == "train"]
valid_df = pd.read_excel(path_labels + 'blue_cards_labels.xlsx', header=0)
valid_df = valid_df[valid_df["split"] == "validation"]

# Specify number of training and validation examples
number_of_examples_train = len(train_df)
number_of_examples_valid = len(valid_df)

train_df = train_df.head(number_of_examples_train)
valid_df = valid_df.head(number_of_examples_valid)

# Reset the indices to start from zero
train_df.reset_index(drop=True, inplace=True)
valid_df.reset_index(drop=True, inplace=True)

print(f'\nTrain data frame: \n{len(train_df)}\n\nValidation data frame: \n{len(valid_df)}\n')
print(f'\nTrain data frame: \n{train_df.head()}\n\nValidation data frame: \n{valid_df.head()}\n')

# Define dataset function for text and image

class CustomDataset(Dataset):
    def __init__(self, root_dir, df, tokenizer, feature_extractor, max_target_length=50):
        self.root_dir = root_dir
        self.df = df
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get file name + text
        file_name = self.df['preproc_file_name'][idx]
        text = self.df['label_eos'][idx]
        # Prepare image (i.e. resize + normalize)
        image = Image.open(self.root_dir + file_name).convert("RGB")
        pixel_values = self.feature_extractor(image, return_tensors="pt").pixel_values
        # Add labels (input_ids) by encoding the text
        labels = self.tokenizer(text,
                                          padding="max_length",
                                          max_length=self.max_target_length).input_ids
        labels = [label if label != self.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding

# Train tokenizer on labels

from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
from datasets import list_datasets, load_dataset
from datasets import Dataset

# Corpus path
corpus_path = tokenizer_path + 'corpus_tokenizer_training.xlsx'
corpus = pd.read_excel(corpus_path, header = None)
dataset = Dataset.from_pandas(corpus)

batch_size = 1000
all_texts = [dataset[i : i + batch_size]["0"] for i in range(0, len(dataset), batch_size)]

def batch_iterator():
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["0"]

# Initialize tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

special_tokens_map = {'cls_token': '[CLS]', 'pad_token': '[PAD]', 'eos_token': '[SEP]', 'unk_token': '[UNK]', }
num_added_toks = tokenizer.add_special_tokens(list(special_tokens_map))

# Train tokenizer
trainer = trainers.BpeTrainer(vocab_size=50, special_tokens=list(special_tokens_map)) # Check recommendations for vocabulary size
tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

# Post-processor and decoder
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False, )
tokenizer.decoder = decoders.ByteLevel()

# Save the tokenizer you trained
tokenizer.save(tokenizer_path + "byte-level-BPE.tokenizer.json")

# Load it using transformers (required, otherwise it is not a callable object)
tokenizer = PreTrainedTokenizerFast(tokenizer_file= tokenizer_path + "byte-level-BPE.tokenizer.json")

from transformers import AutoTokenizer
feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k")

# Uncomment this if you are training your model from the first time (from scratch)

'''
# Load architectures in the model
config_encoder = SwinConfig()
config_decoder = BertConfig()

# Group architectures and define model
config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
model = VisionEncoderDecoderModel(config=config)
'''

# Load architectures in the model
model = VisionEncoderDecoderModel.from_pretrained(model_path)

def  model_size(model):
  return sum(t.numel() for t in model.parameters())

start_size = f'START SIZE:\nSwin size: {model_size(model.encoder)/1000**2:.1f}M parameters\nBERT size: {model_size(model.decoder)/1000**2:.1f}M parameters\nSwin+GPT2 size: {model_size(model)/1000**2:.1f}M parameters\n'

# Write the strings
def build_text_files(data_list, dest_path):
    f = open(dest_path, 'w')
    f.write(data_list)

build_text_files(start_size, output_dir + '/start_size.txt')

tokenizer.pad_token =  '[PAD]'
tokenizer.cls_token = '[CLS]'
tokenizer.model_max_length = 50

# Add add a new classification token
special_tokens_dict = {'cls_token': '[CLS]', 'pad_token': '[PAD]', 'eos_token': '[SEP]', 'unk_token': '[UNK]'}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
print('We have added', num_added_toks, 'tokens')
model.decoder.resize_token_embeddings(len(tokenizer))  # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.

# Initialize the training and evaluation datasets:
train_dataset = CustomDataset(root_dir= '/path/to/train_preprocessed/',
                           df=train_df,
                           feature_extractor = feature_extractor,
                           tokenizer = tokenizer)
eval_dataset = CustomDataset(root_dir= '/path/to/validation_preprocessed/',
                           df=valid_df,
                           feature_extractor = feature_extractor,
                           tokenizer = tokenizer)

# Print number of training and validation examples
print("Number of training examples:", len(train_dataset))
print("Number of validation examples:", len(eval_dataset))

# Verify an example from the training dataset:

encoding = train_dataset[0]
print(encoding)
for k,v in encoding.items():
  print(k, v.shape)

# Check the original image and decode the labels:

image = Image.open(train_dataset.root_dir + train_df['file_name'][0]).convert("RGB")
image.show()

labels = encoding['labels']
labels[labels == -100] = tokenizer.pad_token_id
label_str = tokenizer.decode(labels, skip_special_tokens= True)

# Create dataloaders for training and evaluation:
from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
eval_dataloader = DataLoader(eval_dataset, batch_size=8)

# Specify device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# Make sure vocab size is set correctly
model.config.vocab_size = model.config.decoder.vocab_size

print(model.config.decoder.vocab_size)
print(model.config.decoder_start_token_id)

model.to(device)

# Set beam search parameters
model.config.eos_token_id = tokenizer.eos_token_id
model.config.max_length = 50 
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 100 
model.config.num_beams = 1

final_size = f'AFTER LOADING TOKENIZER:\nSwin size: {model_size(model.encoder)/1000**2:.1f}M parameters\nBERT size: {model_size(model.decoder)/1000**2:.1f}M parameters\nSwin+GPT2 size: {model_size(model)/1000**2:.1f}M parameters\n'

build_text_files(final_size, output_dir + '/end_size.txt')

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import numpy as np

# Define training arguments
epochs = 25
batch_size = 48
eval_steps = np.round(len(train_df)/batch_size*epochs,0)
logging_steps = np.round(eval_steps/10, 0)


training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
#    evaluation_strategy = "epoch",
    evaluation_strategy = "steps",
    num_train_epochs = epochs,    
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    fp16=True, 
    output_dir=output_dir,
    logging_strategy = "steps",
    logging_steps=logging_steps, 
    save_steps=999999,
    eval_steps=eval_steps,  # examples/batchsize*epochs = 33308/48*30 = 3750*12 = 20000
)

from torchmetrics import CharErrorRate

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    metric = CharErrorRate()
    cer = metric(pred_str, label_str)

    return {"cer": cer}

from transformers import default_data_collator

# Instantiate trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=default_data_collator,
)

# Train model
trainer.train()

# Save model
model.save_pretrained(output_dir)

