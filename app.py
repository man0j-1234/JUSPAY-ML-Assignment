# Gmail vs Calendar Query Classifier using BERT (Deployment Version)
# -------------------------------------------------------------------
# This script loads a pre-trained BERT-based model and tokenizer
# to classify queries as Gmail or Calendar. It uses Streamlit for
# an interactive interface.

import os
os.environ["USE_TF"] = "0"  # Disable TensorFlow in Transformers

import pandas as pd
import torch
import re
import calendar
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# --------------------------------------------------------
# CONFIGURATION: Google Drive file IDs for model/tokenizer
# --------------------------------------------------------
MODEL_ZIP_ID = "1bo5akUEzOYdz3OWnw37fleJhSvu6Y_FE"       # saved_model.zip
TOKENIZER_ZIP_ID = "1KxG4prB5Y5s8HOy11wPI37s7tAXU8xyB"   # saved_tokenizer.zip

MODEL_DIR = "saved_model"
TOKENIZER_DIR = "saved_tokenizer"

# --------------------------
# Utility: Download from GDrive
# --------------------------
def download_from_gdrive(file_id, dest_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(url)
    with open(dest_path, 'wb') as f:
        f.write(response.content)

# --------------------------
# Utility: Unzip the model/tokenizer
# --------------------------
def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

# --------------------------
# Load model & tokenizer (only once)
# --------------------------
@st.cache_resource
def load_model_and_tokenizer():
    if not os.path.exists(MODEL_DIR):
        download_from_gdrive(MODEL_ZIP_ID, "model.zip")
        unzip_file("model.zip", MODEL_DIR)

    if not os.path.exists(TOKENIZER_DIR):
        download_from_gdrive(TOKENIZER_ZIP_ID, "tokenizer.zip")
        unzip_file("tokenizer.zip", TOKENIZER_DIR)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model

# --------------------------
# Extract time range if month-year exists in Calendar query
# --------------------------
def extract_date_range(query):
    match = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})", query, re.IGNORECASE)
    if match:
        month = match.group(1).capitalize()
        year = int(match.group(2))
        month_num = list(calendar.month_name).index(month)
        num_days = calendar.monthrange(year, month_num)[1]
        return f"from: {month} 1, {year} to: {month} {num_days}, {year}"
    return None

# Load the tokenizer and model
tokenizer, model = load_model_and_tokenizer()
# --------------------------
# UI
# --------------------------
st.title("Gmail vs Calendar Query Classifier - by Manoj")

user_input = st.text_input("Enter your query:")
if user_input:
    inputs = tokenizer(user_input, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    category = "Gmail" if prediction == 0 else "Calendar"
    st.success(f"Predicted Category: {category}")

    if category == "Calendar":
        time_range = extract_date_range(user_input)
        if time_range:
            st.info(f"Time Range Extracted: {time_range}")
