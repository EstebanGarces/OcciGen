# Setup

#get_ipython().system('pip install -q datasets jiwer')
#get_ipython().system('pip install fastwer')
#get_ipython().system('pip install transformers==4.11.3')
#pip3 install transformers==4.11.3
#!pip install numpy
from transformers import BertConfig, VisionEncoderDecoderModel, BertTokenizer, VisionEncoderDecoderConfig, PreTrainedTokenizerFast, SwinConfig, SwinModel, AutoFeatureExtractor
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt
import wandb

#path = 'C:/Users/valla/OneDrive/Desktop/ocr-project/data_augmentation/' # Adapt this line to your own path
path = '/home/ubuntu/nashome/PrivatLRZ/nashome/PrivatLRZ/'
#path = '/home/ubuntu/'

# Load labels (training)

#data = pd.read_excel(path + 'old2newname.xlsx', header=0)
#train_df, valid_df = train_test_split(data, test_size=0.25, shuffle = False, stratify = None)

train_df = pd.read_excel(path + 'train_40000_aug.xlsx', header=0)
#train_df = pd.read_excel(path + 'data_frame_image_names_aug.xlsx', header=0)
valid_df = pd.read_excel(path + 'test_40000.xlsx', header=0)

from sklearn.model_selection import GroupShuffleSplit 

number_of_examples_train = len(train_df)
number_of_examples_test = len(valid_df)*1

train_df = train_df.head(number_of_examples_train)
valid_df = valid_df.head(number_of_examples_test)

# Reset the indices to start from zero
train_df.reset_index(drop=True, inplace=True)
valid_df.reset_index(drop=True, inplace=True)

print(f'\nTrain data frame: \n{len(train_df)}\n\nValidation data frame: \n{len(valid_df)}\n')

print(f'\nTrain data frame: \n{train_df.head()}\n\nValidation data frame: \n{valid_df.head()}\n')

# Define dataset function for text and image

# Each element of the dataset should return 2 things:
# * `pixel_values`, which serve as input to the model.
# * `labels`, which are the `input_ids` of the corresponding text in the image.

class SyntheticDataset(Dataset):
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
        file_name = self.df['file_name'][idx]
        text = self.df['text_bert'][idx]
        # Prepare image (i.e. resize + normalize)
        image = Image.open(self.root_dir + file_name).convert("RGB")
        pixel_values = self.feature_extractor(image, return_tensors="pt").pixel_values
        # Add labels (input_ids) by encoding the text
        labels = self.tokenizer(text,
                                          padding="max_length",
                                          max_length=self.max_target_length).input_ids
        # Important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding

# Train tokenizer on own corpus

from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
from datasets import list_datasets, load_dataset
from datasets import Dataset


# Corpus path
corpus_path = '/home/ubuntu/nashome/PrivatLRZ/nashome/PrivatLRZ/new_corpus.xlsx'
corpus = pd.read_excel(corpus_path, header = None)
dataset = Dataset.from_pandas(corpus)

batch_size = 1000 # Check effect of modifying this parameter
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

'''
# Sanity check

print(tokenizer.encode("ABCD@FGHIJ<|endoftext|>").ids)
print(tokenizer.decode([0,1,2,3,4,5,423], skip_special_tokens = False))

'''

# Save the tokenizer you trained
tokenizer.save(path + "byte-level-BPE.tokenizer.json")

# Load it using transformers (required, otherwise it is not a callable object)
tokenizer = PreTrainedTokenizerFast(tokenizer_file= path + "byte-level-BPE.tokenizer.json")

# Define processor (wrapping tokenizer and feature extraction from Beit)
from transformers import TrOCRProcessor, AutoTokenizer

feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k")
#pretrained_tokenizer = AutoTokenizer.from_pretrained("gpt2") # Uncomment to use a pre-trained tokenizer
#processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten") # wrap function from TrOCR that combines feature_extractor and tokenizer

'''
# Load architectures in the model
config_encoder = SwinConfig()
config_decoder = BertConfig()

# Group architectures and define model
config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
model = VisionEncoderDecoderModel(config=config)
'''


# Output directory

input_dir = '/home/ubuntu/nashome/PrivatLRZ/nashome/PrivatLRZ/swin_BERT_syn'
output_dir = '/home/ubuntu/nashome/PrivatLRZ/nashome/PrivatLRZ/swin_BERT_syn'
#input_dir = '/home/ubuntu/valhome/PrivatLRZ/swin_BERT_syn'
#output_dir = '/home/ubuntu/valhome/PrivatLRZ/swin_bert_combi'

# Load architectures in the model
model = VisionEncoderDecoderModel.from_pretrained(input_dir)

def  model_size(model):
  return sum(t.numel() for t in model.parameters())

start_size = f'START SIZE:\nSwin size: {model_size(model.encoder)/1000**2:.1f}M parameters\nBERT size: {model_size(model.decoder)/1000**2:.1f}M parameters\nSwin+GPT2 size: {model_size(model)/1000**2:.1f}M parameters\n'

# Write the strings
def build_text_files(data_list, dest_path):
    f = open(dest_path, 'w')
    f.write(data_list)

build_text_files(start_size, output_dir + '/start_size.txt')

#tokenizer.pad_token_id = 'pad_token'
tokenizer.pad_token =  '[PAD]'
tokenizer.cls_token = '[CLS]'
#tokenizer.sep_token = '<SEP>'
#tokenizer.bos_token = '<|endoftext|>'
#tokenizer.eos_token = '<|endoftext|>'
#tokenizer.unk_token = '<UNK>'
#tokenizer.additional_special_tokens = '@'
tokenizer.model_max_length = 50

# Add add a new classification token to GPT-2
special_tokens_dict = {'cls_token': '[CLS]', 'pad_token': '[PAD]', 'eos_token': '[SEP]', 'unk_token': '[UNK]'}

num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
print('We have added', num_added_toks, 'tokens')
model.decoder.resize_token_embeddings(len(tokenizer))  # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.

# Print architecture
#print(config)
#print(model)

#print(f'Own tokenizer: \n{tokenizer} \n\n Pre-trained tokenizer: \n{pretrained_tokenizer}\n')      
#print(f'PAD Own tokenizer: \n{tokenizer.pad_token} \n\n PAD Pre-trained tokenizer: \n{pretrained_tokenizer.pad_token}\n')
#print(f'PAD ID Own tokenizer: \n{tokenizer.pad_token_id} \n\n PAD ID Pre-trained tokenizer: \n{pretrained_tokenizer.pad_token_id}')

# Initialize the training and evaluation datasets:
train_dataset = SyntheticDataset(root_dir= '/home/ubuntu/nashome/PrivatLRZ/nashome/PrivatLRZ/train_preprocessed_aug/',
                           df=train_df,
                           feature_extractor = feature_extractor,
                           tokenizer = tokenizer)
eval_dataset = SyntheticDataset(root_dir= '/home/ubuntu/nashome/PrivatLRZ/nashome/PrivatLRZ/test_preprocessed/',
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

# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
# Make sure vocab size is set correctly
model.config.vocab_size = model.config.decoder.vocab_size

print(model.config.decoder.vocab_size)
print(model.config.decoder_start_token_id)

# model= torch.nn.DataParallel(model) # To use all available GPUs
model.to(device)

# Importantly, we need to set a couple of attributes, namely:
# * the attributes required for creating the `decoder_input_ids` from the `labels` (the model will automatically create the `decoder_input_ids` by shifting the `labels` one position to the right and prepending the `decoder_start_token_id`, as well as replacing ids which are -100 by the pad_token_id)
# * the vocabulary size of the model (for the language modeling head on top of the decoder)
# * beam-search related parameters which are used when generating text.

# Set special tokens used for creating the decoder_input_ids from the labels

# Set beam search parameters
model.config.eos_token_id = tokenizer.eos_token_id
model.config.max_length = 50 # 32 if it were characters
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 100 
#model.config.length_penalty = 2.0
model.config.num_beams = 4

final_size = f'AFTER LOADING TOKENIZER:\nSwin size: {model_size(model.encoder)/1000**2:.1f}M parameters\nBERT size: {model_size(model.decoder)/1000**2:.1f}M parameters\nSwin+GPT2 size: {model_size(model)/1000**2:.1f}M parameters\n'

build_text_files(final_size, output_dir + '/end_size.txt')

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
#    evaluation_strategy = "epoch",
    evaluation_strategy = "steps",
    num_train_epochs=30, # 20      
    per_device_train_batch_size=96, # 48
    per_device_eval_batch_size=96, # 48
    fp16=True, 
    output_dir=output_dir,
    logging_strategy = "steps",
    logging_steps=10000, # There are in total 21000
    save_steps=100000,
    eval_steps=10000,  #examp/batchsize*epochs = 33308/48*30 = 3750*12 = 20000
)

# Evaluate the model on the Character Error Rate (CER), which is available in HuggingFace Datasets (see [here](https://huggingface.co/metrics/cer)).

from datasets import load_metric
cer_metric = load_metric("cer")

# import fastwer

from torchmetrics import CharErrorRate

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    #label_str = "".join(tokenizer.batch_decode(label_ids, skip_special_tokens=True))  
    metric = CharErrorRate()
    cer = metric(pred_str, label_str)

    return {"cer": cer}

from transformers import default_data_collator

# instantiate trainer
trainer = Seq2SeqTrainer(
    model=model,
#    tokenizer=feature_extractor,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=default_data_collator,
)
trainer.train()

#output_dir = '/home/ubuntu/valhome/PrivatLRZ/swin_bert_combi'
model.save_pretrained(output_dir)

