# Email Spam Detection with DistilBERT

A transformer-based binary classifier for spam/ham email detection, fine-tuned on the SpamAssassin dataset using DistilBERT. Achieves 98.8% accuracy on the test set.

Co-authored with Martin Tran as a final project for CS 171 at San Jose State University.

---

## Results

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Spam  | 0.982     | 0.983  | 0.982    |
| Ham   | 0.991     | 0.990  | 0.990    |
| **Overall Accuracy** | | | **98.8%** |

---

## Model

- **Architecture**: `distilbert-base-uncased` with a sequence classification head
- **Dataset**: SpamAssassin (10,749 emails; 35.3% spam, 64.7% ham)
- **Data split**: 72% train / 8% validation / 20% test (stratified)
- **Optimizer**: AdamW, learning rate 1e-5 with linear decay
- **Batch size**: 16
- **Epochs**: 3

DistilBERT was chosen for its reduced size and inference speed relative to full BERT, with minimal accuracy tradeoff for this type of classification task.

---

## Project Structure

```
EmailSpamDetection/
├── main.py                        # Entry point: data loading, splitting, training, evaluation
├── data_pipeline/
│   └── data_pipeline.py           # Email parsing, text cleaning, and dataframe construction
├── model/
│   └── model.py                   # Tokenization, training loop, and test evaluation
├── EmailData/
│   ├── _easy_ham/easy_ham/
│   ├── _hard_ham/hard_ham/
│   ├── _spam/spam/
│   └── _spam_2/spam_2/
└── best_distilbert_spam_classifier.pt   # Saved model weights (generated after first run)
```

---

## Setup

**Requirements**

```bash
pip install torch transformers scikit-learn pandas
```

**Dataset**

Download the SpamAssassin dataset and place the email files in the `EmailData/` directory, matching the folder structure above:

https://huggingface.co/datasets/talby/spamassassin

**Running**

```bash
python main.py
```

On the first run, the model will train and save weights to `best_distilbert_spam_classifier.pt`. Subsequent runs will skip training and go directly to test evaluation.

---

## Data Pipeline

Raw emails are parsed using Python's `email` standard library to extract the subject line and body. The text is then cleaned with the following steps:

1. Quoted-printable soft line breaks (`=\r\n`, `=\n`) are removed
2. All newlines, carriage returns, and tabs are replaced with spaces
3. URLs are replaced with the token `URL`
4. Email addresses are replaced with the token `EMAIL`
5. Consecutive whitespace is collapsed to a single space

The cleaned text is tokenized using the DistilBERT uncased tokenizer with a maximum sequence length of 512 tokens. Shorter sequences are padded.

---

## Training

The training loop runs for 3 epochs. At the end of each epoch, validation loss and a full classification report are printed. The model checkpoint with the lowest validation loss is saved to disk.

Gradient clipping (max norm 1.0) is applied at each step to stabilize training.

A mild uptick in validation loss at epoch 3 was observed, suggesting slight overfitting. Adding early stopping is a straightforward next step.

---

## Potential Improvements

- Early stopping based on validation loss
- Addressing the class imbalance (65/35 ham/spam) through undersampling, oversampling, or merging additional datasets
- Comparison against a non-transformer baseline such as SVM or Naive Bayes
- Extending to multi-class classification (e.g., phishing, marketing, legitimate)

---

## Reference

Short, R. & Tran, M. (2025). *Email Spam Detection and Classification Using Transformer-Based Models.* San Jose State University.
