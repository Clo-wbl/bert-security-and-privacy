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

# Print the content of the XML file
with open("deid_without_ids.xml", "r") as f:
    data = f.read()

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
# print("----------------------------------")
# print("entities to keep : ", entities_to_keep)

# # Fonction pour masquer les entités
# def mask_entities(text, labels_to_mask):
#     for ent in text:
#         if ent.label_ in entities_to_mask:
#             masked_text = masked_text.replace(ent.word, "*****")
#         if pattern_date.match(ent.word):
#             masked_text = masked_text.replace(ent.word, "DATE")
#     return masked_text

# # Mask the entities
# masked_text = mask_entities(ner_results, entities_to_mask)

# # Print the masked text
# print(masked_text)

# masked_xml = nlp(masked_text)
#displacy.serve(masked_xml, style="ent")

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
print("Texte avec les entités masquées :")
print(masked_text[:1000])


# %%
# def mask_entities_in_text(text, entities_to_mask):
#     """
#     Masque les mots correspondants aux entités dans le texte en les remplaçant par des ****.
    
#     :param text: Texte original à modifier.
#     :param entities_to_mask: Liste des entités à masquer, chaque entité est un dictionnaire avec un champ 'word'.
#     :return: Texte modifié avec les entités masquées.
#     """
#     # Parcourir chaque entité et remplacer les mots correspondants dans le texte
#     for entity in entities_to_mask:
#         word = entity.get('word', '').strip()  # Récupérer le mot de l'entité
#         if word:
#             # Remplacer toutes les occurrences du mot par des **** (longueur adaptée)
#             mask = '*' * len(word)
#             text = text.replace(word, mask)
#     return text

# # Appliquer la fonction pour masquer les entités
# masked_text = mask_entities_in_text(record_str, entities_to_mask)

# # Afficher le résultat
# print("Texte avec les entités masquées :")
# print(masked_text[:1000])


# %%
# def mask_entities(text, ner_results, labels_to_mask):
#     # Trier les entités par position de départ pour garantir un ordre correct
#     ner_results_sorted = sorted(ner_results, key=lambda x: x['start'])
    
#     # Initialiser une liste pour le texte reconstruit
#     masked_text = ""
    
#     # Position actuelle dans le texte
#     current_position = 0

#     # Parcourir chaque entité détectée
#     for entity in ner_results_sorted:
#         start = entity['start']
#         end = entity['end']
        
#         # # Ajouter le texte entre la position actuelle et le début de l'entité
#         # if current_position < start:
#         #     masked_text += text[current_position:start]
        
#         # Ajouter "****" si l'entité doit être masquée, sinon ajouter le texte de l'entité
#         if entity['entity'] in labels_to_mask:
#             masked_text += "****"
#         else:
#             masked_text += text[start:end]
        
#         # Mettre à jour la position actuelle
#         current_position = end
    
#     # Ajouter le texte restant après la dernière entité
#     if current_position < len(text):
#         masked_text += text[current_position:]
    
#     return masked_text


# # Texte original
# original_text = "This is a sample text where 12/09/2024 and John's location are mentioned."

# # Appliquer la fonction
# masked_text = mask_entities(original_text, ner_results, labels_to_mask)



