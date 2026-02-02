import torch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW

from data_pipeline.data_pipeline import get_data_as_clean_dataframe


def tokenize_data(texts, labels, max_length=512):
    # Load DistilBERT tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    encodings = tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )

    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']
    labels_tensor = torch.tensor(labels)

    return input_ids, attention_mask, labels_tensor

def train_model(model, epochs, device, train_dataloader, optimizer, scheduler, val_dataloader):
    best_val_loss = float('inf')

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print('-' * 10)

        # Training phase
        model.train()
        total_train_loss = 0

        for batch in train_dataloader:
            batch = tuple(b.to(device) for b in batch)
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'labels': batch[2]
            }

            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        total_val_loss = 0
        val_preds = []
        val_true_labels = []

        with torch.no_grad():
            for batch in val_dataloader:
                batch = tuple(b.to(device) for b in batch)
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'labels': batch[2]
                }

                outputs = model(**inputs)
                loss = outputs.loss
                logits = outputs.logits

                total_val_loss += loss.item()

                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_true_labels.extend(inputs['labels'].cpu().numpy())

        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Validation loss: {avg_val_loss:.4f}")

        # Print validation metrics
        print("\nValidation Results:")
        print(classification_report(val_true_labels, val_preds, target_names=['ham', 'spam'], digits=3))

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_distilbert_spam_classifier.pt')
            print("Model saved!")


def test_model(model, test_dataloader, device):
    # Test the model
    state = torch.load("best_distilbert_spam_classifier.pt", map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    test_preds = []
    test_true_labels = []
    test_probs = []

    with torch.no_grad():
        for batch in test_dataloader:
            batch = tuple(b.to(device) for b in batch)
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'labels': batch[2]
            }

            outputs = model(**inputs)
            logits = outputs.logits

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_true_labels.extend(inputs['labels'].cpu().numpy())

            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()  # Probability for spam
            test_probs.extend(probs)

    print("\nTest Set Results:")
    print(classification_report(test_true_labels, test_preds, target_names=['ham', 'spam'], digits=3))
