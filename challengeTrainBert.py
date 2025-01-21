import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForMaskedLM, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET

# Function to extract and clean text from XML file
def extract_text_from_xml(xml_file):
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Function to extract all text from XML tags
    def get_text_from_element(element):
        if element.text:
            return element.text.strip()
        return ""
    
    # Extract text from each element
    all_text = []
    for elem in root.iter():
        text = get_text_from_element(elem)
        if text:
            all_text.append(text)
    
    # Join the extracted text
    return " ".join(all_text)


# Example usage
xml_file = 'output.xml'  # Path to your XML file
extracted_text = extract_text_from_xml(xml_file)

# Step 1: Prepare the dataset
split_text = extracted_text.split('. ')  # Split the text into sentences
print(f"Length: {len(split_text)}")
print(f"First sentence: {split_text[:5]}")  # Display the first 5 sentences for reference

# Flatten the list of sentences (remove the list wrapper around split_text)
texts = split_text  # This should be a flat list of sentences

# Split the data into training and validation sets
train_texts, val_texts = train_test_split(texts, test_size=2/3, train_size=1/3)

# Step 2: Create a custom Dataset for BERT
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # Tokenize the text and add special tokens
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze(0)  # Remove batch dimension
        attention_mask = encoding['attention_mask'].squeeze(0)
        return {'input_ids': input_ids, 'attention_mask': attention_mask}

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create DataLoader objects
train_dataset = TextDataset(train_texts, tokenizer)
val_dataset = TextDataset(val_texts, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=8)

# Step 3: Load a pre-trained BERT model
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# Step 4: Set up the optimizer, loss function, and scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)

# Total number of training steps (based on batch size and number of epochs)
total_steps = len(train_dataloader) * 3  # Let's assume 3 epochs

# Learning rate scheduler
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Step 5: Train the model

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
for epoch in range(3):  # Loop over the dataset multiple times
    model.train()
    for batch in train_dataloader:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        
        # Compute loss (Masked Language Modeling loss)
        loss = outputs.loss
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Print loss (optional)
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    # Validate the model (optional)
    model.eval()
    total_eval_loss = 0
    for batch in val_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
        
        total_eval_loss += outputs.loss.item()

    print(f"Validation Loss after epoch {epoch+1}: {total_eval_loss / len(val_dataloader)}")

# Step 6: Save the fine-tuned model
model.save_pretrained('fine_tuned_bert')
tokenizer.save_pretrained('fine_tuned_bert')
