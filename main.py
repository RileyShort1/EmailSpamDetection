
#!pip install datasets
#!pip install tokenizers
#!pip install huggingface_hub
#!pip install transformers

from datasets import load_dataset
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
import re

from data_pipeline.data_pipeline import get_data_as_clean_dataframe
from model.model import tokenize_data, train_model

#from transformers import DistilBertTokenizer
#import matplotlib.pyplot as plt
#import seaborn as sns
#from torch.utils.data import DataLoader, TensorDataset
#from transformers import DistilBertForSequenceClassification
#from torch.optim import AdamW
#from transformers import get_linear_schedule_with_warmup
#from torch import nn
#from sklearn.utils import resample
#from sklearn.metrics import roc_curve, roc_auc_score
#from sklearn.metrics import precision_recall_curve, average_precision_score


if __name__ == '__main__':
    df = get_data_as_clean_dataframe()

    print((df['label'] == 0).sum())
    print((df['label'] == 1).sum())


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



