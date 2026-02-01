import os

from sklearn.model_selection import train_test_split
import torch
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
from transformers import DistilBertForSequenceClassification, get_linear_schedule_with_warmup

from data_pipeline.data_pipeline import get_data_as_clean_dataframe
from model.model import test_model, tokenize_data, train_model

if __name__ == '__main__':
    df = get_data_as_clean_dataframe()

    texts = df['text'].to_list()
    labels = df['label'].to_list()

    # Split into train+val and test first (80% train/val, 20% test)
    texts_train_val, texts_test, labels_train_val, labels_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Now split train+val into train and validation (90% train, 10% validation)
    texts_train, texts_val, labels_train, labels_val = train_test_split(
        texts_train_val, labels_train_val, test_size=0.1, random_state=42, stratify=labels_train_val
    )

    print("Train data size = " + str(len(texts_train)))
    print("Validation data size = " + str(len(texts_val)))
    print("Test data size = " + str(len(texts_test)))

    # Process each dataset split
    train_input_ids, train_attention_mask, train_labels = tokenize_data(texts_train, labels_train)
    val_input_ids, val_attention_mask, val_labels = tokenize_data(texts_val, labels_val)
    test_input_ids, test_attention_mask, test_labels = tokenize_data(texts_test, labels_test)

    # 4. Create PyTorch datasets and DataLoaders
    train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
    val_dataset = TensorDataset(val_input_ids, val_attention_mask, val_labels)
    test_dataset = TensorDataset(test_input_ids, test_attention_mask, test_labels)

    batch_size = 16
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

    epochs = 3
    # Set up the optimizer
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # Set up the learning rate scheduler
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # Define the device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if not os.path.exists('best_distilbert_spam_classifier.pt'):
        # Call training function
        train_model(model, epochs, device, train_dataloader, optimizer, scheduler, val_dataloader)

    test_model(model, test_dataloader, device)
