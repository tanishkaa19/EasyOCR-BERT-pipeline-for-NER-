
import torch
import cv2
import easyocr
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertForTokenClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import re
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
train_images = "/Users/tanishka/Desktop/dL_proj/dl_project/images/train_selected"
train_labels = "/Users/tanishka/Desktop/dL_proj/dl_project/groundtruths/train.csv"

# Preprocess Image
def preprocess_image(image_path):
    try:
        image = cv2.imread(image_path)
        resized_image = cv2.resize(image, (512, 512))
        gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_image = clahe.apply(gray)
        _, thresh = cv2.threshold(enhanced_image, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.medianBlur(thresh, 3)
    except Exception as e:
        print(f"Error in preprocessing image {image_path}: {e}")
        return None

def extract_text_from_image(image_path, reader):
    try:
        preprocessed_image = preprocess_image(image_path)
        result = reader.readtext(preprocessed_image, detail=0)
        return ' '.join(result).strip()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None
    
from PIL import Image
import pytesseract

def extract_text_from_image_tesseract(image_path):
    try:
        preprocessed_image = preprocess_image(image_path)
        text = pytesseract.image_to_string(Image.fromarray(preprocessed_image), lang='eng')
        return text.strip()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def process_images_tesseract(image_folder):
    data = []
    file_list = [filename for filename in os.listdir(image_folder) if filename.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]
    for filename in tqdm(file_list, desc=f"Processing {image_folder}", unit="file"):
        image_path = os.path.join(image_folder, filename)
        text = extract_text_from_image_tesseract(image_path)
        data.append([filename, text])
    df = pd.DataFrame(data, columns=['image_name', 'Extracted Text'])
    df_cleaned = df[df['Extracted Text'] != ""]
    return df_cleaned


df_cleaned_train = process_images_tesseract(train_images)
train_df = pd.read_csv(train_labels)
train_df.rename(columns={'image_link': 'image_name'}, inplace=True)
url_prefix = "https://m.media-amazon.com/images/I/"
train_df['image_name'] = train_df['image_name'].str.replace(url_prefix, "", regex=False)
train_df.drop(columns=['group_id'], inplace=True)
merged_df = pd.merge(train_df, df_cleaned_train, on='image_name')
merged_df.to_csv("/Users/tanishka/Desktop/dL_proj/final1.csv",index=False)

quit()
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'(\d),(\d)', r'\1\2', text)  
        text = re.sub(r'[^\w\s.$]', '', text)  
        return ' '.join(text.split())  
    return ''

# Convert Labels
LABEL_MAP = {"O": 0, "B": 1, "I": 2} 
def label_to_ids(labels):
    return [LABEL_MAP[label] for label in labels]

# Tokenization and Labeling
import pandas as pd
from transformers import BertTokenizer

# def create_token_label_pairs(text, entity_value, output_csv="token_label_pairs.csv"):
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#     tokens = tokenizer.tokenize(text)
#     entity_value_tokens = tokenizer.tokenize(entity_value)

#     # List to store the token-label pairs
#     labels = []
#     # entity_started = False

#     # Process the tokens and assign labels
#     for token in tokens:
#         if token in entity_value_tokens :
#             labels.append("B")  # Beginning of entity
#             # entity_started = True
#             entity_value_tokens.pop(0)
#         # elif token in entity_value_tokens:
#         #     labels.append("I")  # Inside the entity
#         #     entity_value_tokens.pop(0)
#         else:
#             labels.append("O")  # Outside the entity

#     # Create a DataFrame to save the token-label pairs
#     df = pd.DataFrame({
#         'Token': tokens,
#         'Label': labels
#     })

#     # Save to CSV
#     df.to_csv(output_csv, index=False, mode='a', header=not pd.io.common.file_exists(output_csv))
#     # print(f"Token-label pairs saved to {output_csv}")

#     return tokens, labels

# Tokenization and Labeling
def create_sentence_label_pairs(sentences, keywords):
    data = []
    for sentence, keyword_list in zip(sentences, keywords):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Tokenize sentence
        tokens = tokenizer.tokenize(sentence)
        labels = ['O'] * len(tokens)

        # Tokenize and label for each keyword in the sentence
        for keyword in keyword_list:
            keyword_tokens = tokenizer.tokenize(keyword)
            for i in range(len(tokens) - len(keyword_tokens) + 1):
                if tokens[i:i + len(keyword_tokens)] == keyword_tokens:
                    # Assign 'B' to the first token of the keyword, 'I' to the rest
                    labels[i] = 'B'
                    for j in range(1, len(keyword_tokens)):
                        labels[i + j] = 'B'
        
        # Append the sentence and its corresponding labels to the dataset
        data.append({
            'train_sentences': sentence,
            'label': labels
        })
        
    return data


class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        tokens = str(self.texts[idx])
        labels = self.labels[idx]

        encoding = self.tokenizer(
            tokens,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            # return_attention_mask=True,
            return_tensors="pt",
        )
        # print(encoding['input_ids'])
        # print("Vocabulary size:", self.tokenizer.vocab_size)

        labels = labels[:self.max_len] + [LABEL_MAP["O"]] * (self.max_len - len(labels))
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

# Training and Evaluation
def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=10):
    best_val_loss = float('inf')
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            # logits = outputs.logits  # Extract logits from the model output

            loss = criterion(outputs.logits.view(-1, len(LABEL_MAP)), labels.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask,labels)

                logits = outputs.logits 
                logits = logits.view(-1, logits.size(-1))
                labels = labels.view(-1)
                # print(f"logits shape: {logits.shape}")  # Should match [batch_size * seq_len, num_classes]
                # print(f"labels shape: {labels.shape}") 
                loss = criterion(logits,labels)
                val_loss += loss.item()
                epoch_val_loss = val_loss/len(val_loader)

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Epoch {epoch+1}, Training Loss: {total_loss/len(train_loader):.4f}')
        print(f'Validation Loss: {epoch_val_loss:.4f}')

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), 'best_bert_model.pth')

def test_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    predictions = []
    actuals = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            # Convert outputs to predictions (assuming binary classification)
            # predicted = (outputs.squeeze(-1) >= 0.5).float()
            
            # Calculate accuracy
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()
            
            # Store predictions and actuals for later analysis
            # predictions.extend(predicted.cpu().numpy())
            actuals.extend(labels.cpu().numpy())
    
    # Calculate average loss and accuracy
    avg_loss = test_loss / len(test_loader)
    # accuracy = (correct / total) * 100
    
    print(f"Test Loss: {avg_loss:.4f}")
    # print(f"Test Accuracy: {accuracy:.2f}%")
    
    return predictions, actuals, avg_loss
        # print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")

# Prepare Data
def prepare_and_train():
    merged_df['Extracted Text'] = merged_df['Extracted Text'].apply(clean_text)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    texts = merged_df['Extracted Text'].values
    labels = merged_df['entity_value'].values  

    # Tokenize and label using the new function
    tokens_list = []
    label_list = []
    for text, label in zip(texts, labels):
        # Create sentence-label pairs for each text and entity_value
        tokens, token_labels = create_sentence_label_pairs([text], [str(label)])
        tokens_list.append(tokens[0]['train_sentences'])  # Get the sentence
        label_list.append(label_to_ids(token_labels[0]['label']))  # Get the labels

    train_tokens, temp_tokens, train_labels, temp_labels = train_test_split(tokens_list, label_list, test_size=0.3, random_state=42)
    val_tokens, test_tokens, val_labels, test_labels = train_test_split(temp_tokens, temp_labels, test_size=0.5, random_state=42)

    train_dataset = CustomDataset(train_tokens, train_labels, tokenizer)
    val_dataset = CustomDataset(val_tokens, val_labels, tokenizer)
    test_dataset = CustomDataset(test_tokens, test_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)

    model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=len(LABEL_MAP))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss()

    train_model(model, train_loader, val_loader, optimizer, criterion)
    test_model(model, test_loader, criterion, device)

prepare_and_train()