import pandas as pd
from pathlib import Path
import re
from email import policy
from email.parser import BytesParser

EMAIL_FILE = re.compile(r"^\d+\.")

def extract_subject_body(raw_bytes: bytes) -> str:
    msg = BytesParser(policy=policy.default).parsebytes(raw_bytes)

    subject = msg.get("subject") or ""

    # Prefer plain text parts, fall back to whatever exists
    body_parts = []
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = part.get_content_disposition()
            if disp == "attachment":
                continue
            if ctype in ("text/plain", "text/html"):
                try:
                    body_parts.append(part.get_content())
                except Exception:
                    pass
    else:
        try:
            body_parts.append(msg.get_content())
        except Exception:
            pass

    body = "\n".join(body_parts)

    return f"SUBJECT: {subject}\n\n{body}".strip()

def load_folder(path, label, category):
    rows = []
    for file in Path(path).iterdir():
        if file.is_file() and EMAIL_FILE.match(file.name):
            raw = file.read_bytes()
            model_text = extract_subject_body(raw)
            rows.append({
                "text": model_text,
                "label": label,
                "category": category
            })
    return rows

def clean_data(text):
    text = text.replace('=\r\n', '').replace('=\n', '')
    text = text.replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'http\S+|www\.\S+', ' URL ', text)  # Replace URLs with a token
    text = re.sub(r'\S+@\S+', ' EMAIL ', text)  # Replace emails with a token
    text = re.sub(r'\s+', ' ', text).strip()  # Replace multiple spaces with a single space and trim

    return text

def get_data_as_clean_dataframe():
    data = []
    data += load_folder("./EmailData/_easy_ham/easy_ham", 0, "easy_ham")
    data += load_folder("./EmailData/_hard_ham/hard_ham", 0, "hard_ham")
    data += load_folder("./EmailData/_spam/spam", 1, "spam")
    data += load_folder("./EmailData/_spam_2/spam_2", 1, "spam_2")

    df = pd.DataFrame(data)

    df['text'] = df['text'].apply(clean_data)

    return df
