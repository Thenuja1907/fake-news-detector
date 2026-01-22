import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import os

# --- 1. Dataset Class for Transformers ---
class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            add_special_tokens=True,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def train_system():
    print("Initializing RoBERTa Training (Proposal: Fine-tuned RoBERTa)...")
    
    # 1. Load Data
    data_path = "../data/processed/cleaned_data.csv"
    if not os.path.exists(data_path):
        print(f"⚠ Warning: Data not found at {data_path}. Creating dummy data for demonstration.")
        # Create dummy data if file doesn't exist so the user can at least run the script
        df = pd.DataFrame({
            'text': ["This is a fake news article.", "The president gave a speech today.", "Aliens landed in New York!", "Stock market hits record high."],
            'label': [1, 0, 1, 0] # 1 = Fake, 0 = Real (Adjust mapping as needed)
        })
    else:
        df = pd.read_csv(data_path).dropna()

    # Map labels if they are strings
    if df['label'].dtype == object:
        df['label'] = df['label'].map({'REAL': 0, 'FAKE': 1})

    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

    # 2. Tokenizer (RoBERTa)
    print("Loading RoBERTa Tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    train_dataset = FakeNewsDataset(X_train.to_numpy(), y_train.to_numpy(), tokenizer)
    test_dataset = FakeNewsDataset(X_test.to_numpy(), y_test.to_numpy(), tokenizer)

    # 3. Model (RoBERTa)
    print("Loading Pre-trained RoBERTa Model...")
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,              # Low epochs for demo speed
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        no_cuda=True if not torch.cuda.is_available() else False # Use CPU if no GPU
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )

    # 6. Train using RoBERTa
    print("Starting Training (This may take a while)...")
    trainer.train()

    # 7. Save Model
    print("Saving Fine-Tuned Model...")
    save_path = "models/roberta_fake_news"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print("-" * 30)
    print(f"✓ SUCCESS: RoBERTa Model Trained & Saved to {save_path}")
    print("-" * 30)

if __name__ == "__main__":
    train_system()