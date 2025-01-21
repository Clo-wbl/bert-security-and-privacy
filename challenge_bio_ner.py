# %% [markdown]
# ### General Requirements 

# %%
# %pip install beautifulsoup4
# %pip install lxml
# %pip install spacy
# %pip install ipywidgets
# %pip install transformers
# %pip install nlp

# %% [markdown]
# # Prepare the dataset

# %% [markdown]
# ### Open the original dataset

# %%
from bs4 import BeautifulSoup


# Reading the data inside the xml
# file to a variable under the name
# data
with open('deid_surrogate_train_all_version2.xml', 'r') as f:
    data = f.read()

# Passing the stored data inside
# the beautifulsoup parser, storing
# the returned object
Bs_data = BeautifulSoup(data, "xml")

# Using find() to extract attributes
# of the first instance of the tag
b_type = Bs_data.find_all('PHI', {'TYPE':'HOSPITAL'})

# print(b_type)

# %% [markdown]
# ### Reduce dataset

# %%
# Find all recordings
records = Bs_data.find_all("RECORD")

# Divide in 2
mid_index = len(records) // 2
less_records = records[:mid_index]

# Create a new xml
new_root = Bs_data.new_tag("ROOT")
for record in less_records:
    new_root.append(record)

record_test = new_root

# %% [markdown]
# ### Remove IDs if needed

# %%
def remove_ids(soup):
    for item in soup.find_all(attrs={"TYPE": "ID"}):
        item.string = "*****"
    return soup

# %% [markdown]
# ### Remove IDs and labels

# %%
# record_test = remove_ids(record_test) # Remove IDs from the XML data
# record_text = record_test.get_text()
# record_str = str(record_text)
# print(record_str)

# %% [markdown]
# ### Remove IDs but keep labels

# %%
record_test = remove_ids(record_test) # Remove IDs from the XML data
record_str = str(record_test)

# %% [markdown]
# ### Retransform to xml file

# %%
import xml.etree.ElementTree as ET

# Parse the XML string
root = ET.fromstring(record_str)

# Create an ElementTree object
tree = ET.ElementTree(root)

# Write the ElementTree object to an XML file
tree.write("deid_without_ids.xml", encoding="utf-8", xml_declaration=True)

# # Print the content of the XML file
# with open("deid_without_ids.xml", "r") as f:
#     data = f.read()

# %% [markdown]
# ### Split texts

# %%
# Function to split text into manageable chunks
def split_text(text, max_length):
    tokens = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for token in tokens:
        token_length = len(token) + 1  # Add 1 for the space
        if current_length + token_length <= max_length:
            current_chunk.append(token)
            current_length += token_length
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [token]
            current_length = token_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Split the text into chunks
max_chunk_length = 512  # Adjust based on the model's capacity
chunks = split_text(record_str, max_chunk_length)

# %% [markdown]
# # Bio NER

# %% [markdown]
# ### Load the specialized model

# %%
from transformers import AutoModelForTokenClassification

bio_ner_model = AutoModelForTokenClassification.from_pretrained("blaze999/Medical-NER")

# %% [markdown]
# ### Load the tokenizer

# %%
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("blaze999/Medical-NER")

# %% [markdown]
# ### Create an instance of pipeline with the model and the tokenizer

# %%
from transformers import pipeline

ner_pipe = pipeline("ner", model=bio_ner_model, tokenizer=tokenizer)

# %% [markdown]
# ### Extract entities

# %%
# Apply the NER pipeline to each chunk
ner_results = []
for chunk in chunks:
    ner_results.extend(ner_pipe(chunk))

# %%
print(ner_results[:10])

# %% [markdown]
# ### Mask choosen entities

# %%

import re
# Define the entities to mask
labels_to_mask = ["I-AREA", "B-AREA",
                     "B-DATE", "I-DATE",
                       "I-NONBIOLOGICAL_LOCATION", "B-NONBIOLOGICAL_LOCATION",
                         "I-OCCUPATION", "B-OCCUPATION",
                         "I-PERSONAL_BACKGROUND", "B-PERSONAL_BACKGROUND",
                         ]
pattern_date = re.compile("[0-9]{2}\/[0-9]{2}\/[0-9]{2,4}")

entities_to_mask = []
entities_to_keep = []

def find_entities_to_mask(text, labels_to_mask, entities_to_mask, entities_to_keep):
    for ent in text:
        if  ent['entity'] in labels_to_mask:
            entities_to_mask.append(ent)
        else:
            entities_to_keep.append(ent)

find_entities_to_mask(ner_results, labels_to_mask, entities_to_mask, entities_to_keep)

print("entities to mask : ", entities_to_mask)

# %%
def mask_entities_by_tokens(text, entities_to_mask):
    """
    Masque les mots dans le texte si leurs tokens correspondent aux entités à masquer.
    
    :param text: Texte original à modifier.
    :param entities_to_mask: Liste des entités à masquer, chaque entité est un dictionnaire avec un champ 'word'.
    :return: Texte modifié avec les entités masquées.
    """
    # Créer un ensemble des mots à masquer pour une recherche rapide
    words_to_mask = {entity['word'].strip() for entity in entities_to_mask if 'word' in entity}
    
    # Tokeniser le texte par des espaces (simple tokenisation)
    tokens = text.split()
    
    # Masquer les tokens correspondant aux mots à masquer
    masked_tokens = [
        '*' * len(token) if token in words_to_mask else token
        for token in tokens
    ]
    
    # Rejoindre les tokens masqués pour reformer le texte
    return ' '.join(masked_tokens)

# Appliquer la fonction pour masquer les entités
masked_text = mask_entities_by_tokens(record_str, entities_to_mask)

# Afficher le résultat
print("Text with masked entities")
print(masked_text[:1000])


# %% [markdown]
# ### Tranform back to XML

# %%
import xml.etree.ElementTree as ET

# Parse the XML string
root = ET.fromstring(masked_text)

# Create an ElementTree object
tree = ET.ElementTree(root)

# Write the ElementTree object to an XML file
tree.write("deid_masked.xml", encoding="utf-8", xml_declaration=True)

# %% [markdown]
# # Finetuning BERT

# %%
# %pip install datasets evaluate transformers[sentencepiece]
# %pip install accelerate
# To run the training on TPU, you will need to uncomment the following line:
# !pip install cloud-tpu-client==0.10 torch==1.9.0 https://storage.googleapis.com/tpu-pytorch/wheels/torch_xla-1.9-cp37-cp37m-linux_x86_64.whl
# !apt install git-lfs

# %%

from bs4 import BeautifulSoup
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import torch
import collections
import numpy as np
from transformers import default_data_collator
import math

# %% [markdown]
# ## Step 1: Parse the XML file with BeautifulSoup

# %%
def parse_xml(file_path):
    with open(file_path, 'r') as f:
        data = f.read()

    bs_data = BeautifulSoup(data, "lxml-xml")  # Use lxml-xml parser

    # Print the structure of the XML to verify tag names
    # print(bs_data.prettify())

    parsed_data = []
    for item in bs_data.find_all('RECORD'):  # Adjust the tag name as per your XML structure
        text = item.find('TEXT').text
        label = item.find("SMOKING",).get("STATUS")
        parsed_data.append({'text': text, 'label': label})

    return parsed_data

# %%
# Step 2: Convert to Dictionary Format
def convert_to_dict_format(data):
    dict_format = {'text': [], 'label': []}
    for entry in data:
        dict_format['text'].append(entry['text'])
        dict_format['label'].append(entry['label'])
    return dict_format

# %%
# Step 3: Create a dataset from the parsed data
def create_dataset(data):
    return Dataset.from_dict(data)

# File path to your XML file
file_path = 'deid_masked.xml'

# Parse the XML file
parsed_data = parse_xml(file_path)
print(f"Parsed data size: {len(parsed_data)}")  # Debugging statement

# Convert to dictionary format
dict_format_data = convert_to_dict_format(parsed_data)
print(f"Dictionary format data size: {len(dict_format_data['text'])}")  # Debugging statement

# Create a dataset
dataset = create_dataset(dict_format_data)

# Check the dataset size
print(f"Dataset size: {len(dataset)}")

# Shuffle and select samples
dataset_size = len(dataset)
if dataset_size > 0:
    sample = dataset.shuffle(seed=42).select(range(min(3, dataset_size)))

    # Print the samples
    for row in sample:
        # print(f"\n'>>> Review: {row['text']}'")
        print(f"'>>> Label: {row['label']}'")
else:
    print("Dataset is empty.")


# %% [markdown]
# ## Split the dataset into train and test

# %%
if dataset_size > 0:
    train_test_split = dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split['train']
    test_dataset = train_test_split['test']

    # Step 4: Tokenize the dataset
    model_checkpoint = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    def tokenize_function(examples):
        result = tokenizer(examples["text"], padding=True, truncation=True)
        # if tokenizer.is_fast:
        #     result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        return result

    # Use batched=True to activate fast multithreading!
    tokenized_datasets = DatasetDict({
        'train': train_dataset.map(tokenize_function, batched=True, remove_columns=["text", "label"]),
        'test': test_dataset.map(tokenize_function, batched=True, remove_columns=["text", "label"])
    })


    # Step 5: Group texts into chunks
    chunk_size = 128

    def group_texts(examples):
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        # Compute length of concatenated texts
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the last chunk if it's smaller than chunk_size
        total_length = (total_length // chunk_size) * chunk_size
        # Split by chunks of max_len
        result = {
            k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
            for k, t in concatenated_examples.items()
        }
        # Create a new labels column
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(group_texts, batched=True)

    # Step 6: Fine-tune the model
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        remove_unused_columns=False,  # Ensure this is set to False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["test"],
        data_collator=data_collator,
    )
    eval_results = trainer.evaluate()
    print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    trainer.train()
    # Evaluate the model
    eval_results = trainer.evaluate()
    print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
else:
    print("Dataset is empty. Cannot proceed with training.")


