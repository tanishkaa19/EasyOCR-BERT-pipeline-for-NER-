# import torch
# import cv2
# import easyocr
# import os
# import numpy as np
# from tqdm import tqdm
# from transformers import BertTokenizer, BertModel
# import pandas as pd
# from sklearn.model_selection import train_test_split
# import re
# from torch.utils.data import Dataset, DataLoader
# import torch.nn as nn

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# train_images = "/Users/tanishka/Desktop/dL_proj/dl_project/images/train_selected"
# train_labels = "/Users/tanishka/Desktop/dL_proj/dl_project/groundtruths/train.csv"
# test_images = "/Users/tanishka/Desktop/dL_proj/dl_project/images/test_selected"
# test_labels = "/Users/tanishka/Desktop/dL_proj/dl_project/groundtruths/test.csv"

# def preprocess_image(image_path):
#     try:
#         image = cv2.imread(image_path)
#         resizing = cv2.resize(image, (512, 512))
#         gray = cv2.cvtColor(resizing, cv2.COLOR_BGR2GRAY)
#         clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#         enhanced_image = clahe.apply(gray)
#         _, thresh = cv2.threshold(enhanced_image, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#         processed_image = cv2.medianBlur(thresh, 3)
#         return processed_image
#     except Exception as e:
#         print(f"Error in preprocessing image {image_path}: {e}")
#         return None

# import pytesseract
# from PIL import Image

# # Specify the path to Tesseract executable if it's not in your PATH
# # For example, on Windows: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# def extract_text_from_image_tesseract(image_path):
#     try:
#         preprocessed_image = preprocess_image(image_path)
#         text = pytesseract.image_to_string(Image.fromarray(preprocessed_image), lang='eng')
#         return text.strip()
#     except Exception as e:
#         print(f"Error processing {image_path}: {e}")
#         return None

# def process_images_tesseract(image_folder):
#     data = []
#     file_list = [filename for filename in os.listdir(image_folder) if filename.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]
#     for filename in tqdm(file_list, desc=f"Processing {image_folder}", unit="file"):
#         image_path = os.path.join(image_folder, filename)
#         text = extract_text_from_image_tesseract(image_path)
#         data.append([filename, text])
#     df = pd.DataFrame(data, columns=['image_name', 'Extracted Text'])
#     df_cleaned = df[df['Extracted Text'] != ""]
#     return df_cleaned

# df_cleaned_train = process_images_tesseract(train_images)
# print(df_cleaned_train)
# quit()
# train_df = pd.read_csv(train_labels)
# train_df.rename(columns={'image_link': 'image_name'}, inplace=True)
# url_prefix = "https://m.media-amazon.com/images/I/"
# train_df['image_name'] = train_df['image_name'].str.replace(url_prefix, "", regex=False)
# train_df.drop(columns=['group_id'], inplace=True)
# merged_df = pd.merge(train_df, df_cleaned_train, on='image_name')
# merged_df['Extracted Text'] = merged_df['entity_name'] + ': [' + merged_df['Extracted Text'] + ']'
# final_df = merged_df[['Extracted Text', 'entity_value']]
# final_df.to_csv("/Users/tanishka/Desktop/dL_proj/final.csv", index=False)

# # Function to clean text
# def cleaning(text):
#     if isinstance(text, str):
#         # Standardize number format
#         text = re.sub(r'(\d),(\d)', r'\1\2', text)  # Remove commas in numbers
#         # Remove only specific characters while preserving numerical information
#         text = re.sub(r'[^\w\s.$]', '', text)
#         # Standardize spacing
#         text = ' '.join(text.split())
#         return text
#     return ''

# final_df.loc['Extracted Text'] = final_df['Extracted Text'].apply(cleaning)

# # Dataset class for tokenizing text and labels
# class customDataset(Dataset):
#     def __init__(self, texts, labels, tokenizer, max_len=128):
#         self.texts = texts
#         self.labels = labels
#         self.tokenizer = tokenizer
#         self.max_len = max_len
    
#     def __len__(self):
#         return len(self.texts)
    
#     def __getitem__(self, idx):
#         text = str(self.texts[idx])
#         label = self.labels[idx]
        
#         text_encoding = self.tokenizer.encode_plus(
#             text,
#             add_special_tokens=True,
#             max_length=self.max_len,
#             padding='max_length',
#             truncation=True,
#             return_attention_mask=True,
#             return_tensors='pt'
#         )

#         label_tensor = torch.tensor(label, dtype=torch.float32)
#         return {
#             'input_ids': text_encoding['input_ids'].flatten(),
#             'attention_mask': text_encoding['attention_mask'].flatten(),
#             'labels': label_tensor
#         }

# # BERT model architecture
# class BERT_Arch(nn.Module):
#     def __init__(self, bert, tokenizer):
#         super(BERT_Arch, self).__init__()
#         self.bert = bert
#         self.tokenizer = tokenizer
#         self.dropout = nn.Dropout(0.1)
#         self.fc1 = nn.Linear(768, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 1)
#         self.relu = nn.ReLU()

#     def forward(self, sent_id, mask):
#         _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
#         x = self.fc1(cls_hs)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.relu(x)
#         x = self.dropout(x)
#         x = self.fc3(x)
#         return x

# # Training function
# def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda'):
#     model = model.to(device)
#     best_val_loss = float('inf')
#     correct=0.0

#     for epoch in range(num_epochs):
#         model.train()
#         train_loss = 0
#         for batch in train_loader:
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['labels'].to(device)

#             optimizer.zero_grad()
#             outputs = model(input_ids, attention_mask)
#             outputs=outputs.float()

#             loss = criterion(outputs.squeeze(-1), labels)
#             loss.backward()
#             optimizer.step()
#             train_loss += loss.item()
#             epoch_loss = train_loss/len(train_loader)

#             _, predicted = torch.max(outputs.data, 1)
#             correct += (predicted == labels).float().sum()
#             accuracy = 100 * correct / len(train_loader.dataset)

#         # Validation phase
#         model.eval()
#         val_loss = 0
#         with torch.no_grad():
#             for batch in val_loader:
#                 input_ids = batch['input_ids'].to(device)
#                 attention_mask = batch['attention_mask'].to(device)
#                 labels = batch['labels'].to(device)

#                 outputs = model(input_ids, attention_mask)
#                 loss = criterion(outputs.squeeze(-1), labels)
#                 val_loss += loss.item()
#                 epoch_val_loss = val_loss/len(val_loader)

#         print(f'Epoch {epoch + 1}/{num_epochs}:')
#         print(f'Epoch {epoch+1}, Training Loss: {epoch_loss:.4f}')
#         print(f'Validation Loss: {epoch_val_loss:.4f}')

#         if epoch_val_loss < best_val_loss:
#             best_val_loss = epoch_val_loss
#             torch.save(model.state_dict(), 'best_bert_model.pth')
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import train_test_split
# import pandas as pd
# import re
# def test_model(model, test_loader, criterion, device):
#     """
#     Test the model on the test dataset
    
#     Args:
#         model: The trained BERT model
#         test_loader: DataLoader containing test data
#         criterion: Loss function
#         device: Device to run the model on
    
#     Returns:
#         dict: Dictionary containing test metrics
#     """
#     model.eval()
#     test_loss = 0
#     predictions = []
#     actuals = []
    
#     with torch.no_grad():
#         for batch in test_loader:
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['labels'].to(device)
            
#             outputs = model(input_ids, attention_mask)
#             loss = criterion(outputs.squeeze(-1), labels)
#             test_loss += loss.item()
            
#             predictions.extend(outputs.squeeze(-1).cpu().numpy())
#             actuals.extend(labels.cpu().numpy())
    
#     # Calculate average test loss
#     avg_test_loss = test_loss / len(test_loader)
    
#     # Calculate MSE
#     mse = np.mean((np.array(predictions) - np.array(actuals)) ** 2)
    
#     # Calculate RMSE
#     rmse = np.sqrt(mse)
    
#     # Calculate MAE
#     mae = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
    
#     # Calculate R2 score
#     ss_res = np.sum((np.array(actuals) - np.array(predictions)) ** 2)
#     ss_tot = np.sum((np.array(actuals) - np.mean(np.array(actuals))) ** 2)
#     r2 = 1 - (ss_res / ss_tot)
    
#     return {
#         'test_loss': avg_test_loss,
#         'mse': mse,
#         'rmse': rmse,
#         'mae': mae,
#         'r2': r2
#     }

# def prepare_and_train():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")

#     # Load and preprocess data
#     df = pd.read_csv('/Users/tanishka/Desktop/dL_proj/final.csv')
#     df['entity_value'] = df['entity_value'].apply(
#         lambda x: float(re.sub(r'[^\d.]', '', x.split()[0])) if isinstance(x, str) else x
#     )
    
#     # Scale the values
#     scaler = MinMaxScaler()
#     df['scaled_entity_value'] = scaler.fit_transform(df[['entity_value']])

#     # Split data into train, validation, and test sets
#     train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
#         df['Extracted Text'].values,
#         df['entity_value'].values,
#         test_size=0.2,
#         random_state=42
#     )
    
#     train_texts, val_texts, train_labels, val_labels = train_test_split(
#         train_val_texts,
#         train_val_labels,
#         test_size=0.25,
#         random_state=42
#     )

#     # Initialize tokenizer and model
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     bert = BertModel.from_pretrained('bert-base-uncased')

#     # Create datasets
#     train_dataset = customDataset(train_texts, train_labels, tokenizer)
#     val_dataset = customDataset(val_texts, val_labels, tokenizer)
#     test_dataset = customDataset(test_texts, test_labels, tokenizer)
    
#     # Create dataloaders
#     train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=16)
#     test_loader = DataLoader(test_dataset, batch_size=16)

#     # Freeze BERT parameters
#     for param in bert.parameters():
#         param.requires_grad = False

#     # Initialize model and training components
#     model = BERT_Arch(bert, tokenizer)
#     model = model.to(device)
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

#     # Train the model
#     print("\nStarting training...")
#     train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=device)
    
#     # Load best model for testing
#     model.load_state_dict(torch.load('best_bert_model.pth'))
    
#     # Test the model
#     print("\nTesting the model...")
#     test_metrics = test_model(model, test_loader, criterion, device)
    
#     # Print test results
#     print("\nTest Results:")
#     print(f"Test Loss: {test_metrics['test_loss']:.4f}")
#     print(f"MSE: {test_metrics['mse']:.4f}")
#     print(f"RMSE: {test_metrics['rmse']:.4f}")
#     print(f"MAE: {test_metrics['mae']:.4f}")
#     print(f"R2 Score: {test_metrics['r2']:.4f}")
    
#     return model, test_metrics, scaler

# if __name__ == "__main__":
#     model, test_metrics, scaler = prepare_and_train()



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

def process_images(image_folder):
    reader = easyocr.Reader(['en'], gpu=True)
    data = []
    file_list = [filename for filename in os.listdir(image_folder) if filename.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]
    for filename in tqdm(file_list, desc=f"Processing {image_folder}", unit="file"):
        image_path = os.path.join(image_folder, filename)
        text = extract_text_from_image(image_path, reader)
        data.append([filename, text])
    df = pd.DataFrame(data, columns=['image_name', 'Extracted Text'])
    df_cleaned = df[df['Extracted Text'] != ""]
    return df_cleaned


df_cleaned_train = process_images(train_images)
train_df = pd.read_csv(train_labels)
train_df.rename(columns={'image_link': 'image_name'}, inplace=True)
url_prefix = "https://m.media-amazon.com/images/I/"
train_df['image_name'] = train_df['image_name'].str.replace(url_prefix, "", regex=False)
train_df.drop(columns=['group_id'], inplace=True)
merged_df = pd.merge(train_df, df_cleaned_train, on='image_name')
merged_df.to_csv("/Users/tanishka/Desktop/dL_proj/final1.csv",index=False)

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

def create_token_label_pairs(text, entity_value, output_csv="token_label_pairs.csv"):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    tokens = tokenizer.tokenize(text)
    entity_value_tokens = tokenizer.tokenize(entity_value)

    # List to store the token-label pairs
    labels = ['O'] * len(tokens)

    # Assign 'B' for every token in the entity (no 'I' labels used)
    for i in range(len(tokens) - len(entity_value_tokens) + 1):
        if tokens[i:i + len(entity_value_tokens)] == entity_value_tokens:
            for j in range(len(entity_value_tokens)):
                labels[i + j] = 'B'

    # Create a DataFrame to save the token-label pairs
    df = pd.DataFrame({
        'Token': tokens,
        'Label': labels
    })

    # Save to CSV
    df.to_csv(output_csv, index=False, mode='a', header=not pd.io.common.file_exists(output_csv))

    return tokens, labels


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

# Model Definition
# class BERT_Arch(nn.Module):
#     def __init__(self, bert, num_labels):
#         super(BERT_Arch, self).__init__()
#         self.bert = bert
#         self.dropout = nn.Dropout(0.1)
#         self.fc = nn.Linear(768, num_labels)
    
#     def forward(self, input_ids, attention_mask):
#         outputs = self.bert(input_ids, attention_mask=attention_mask)
#         output = self.dropout(outputs[0])
#         return self.fc(output)

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

from sklearn.metrics import classification_report

def test_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            # Get predicted labels
            _, predicted = torch.max(outputs.logits, dim=-1)
            predictions.extend(predicted.cpu().numpy())
            actuals.extend(labels.cpu().numpy())
    
    # Calculate average loss
    avg_loss = test_loss / len(test_loader)
    
    # Flatten the lists for classification report
    predictions_flat = [item for sublist in predictions for item in sublist]
    actuals_flat = [item for sublist in actuals for item in sublist]
    
    # Generate the classification report
    report = classification_report(actuals_flat, predictions_flat, target_names=[str(i) for i in range(len(LABEL_MAP))])
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Classification Report:\n{report}")
    
    return predictions_flat, actuals_flat, avg_loss

# Prepare Data
def prepare_and_train():
    merged_df['Extracted Text'] = merged_df['Extracted Text'].apply(clean_text)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    texts = merged_df['Extracted Text'].values
    labels = merged_df['entity_value'].values  
    # Tokenize and label
    tokens_list = []
    label_list = []
    for text, label in zip(texts, labels):
        tokens, token_labels = create_token_label_pairs(text, str(label))
        tokens_list.append(tokens)
        label_list.append(label_to_ids(token_labels))

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

# Execute Training
prepare_and_train()