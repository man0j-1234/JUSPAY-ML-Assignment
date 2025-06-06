# Gmail vs Calendar Query Classifier using BERT (Deployment Version)
# -------------------------------------------------------------------
# This script loads a pre-trained BERT-based model and tokenizer
# to classify queries as Gmail or Calendar. It uses Streamlit for
# an interactive interface.

import os
os.environ["USE_TF"] = "0"  # Disable TensorFlow in Transformers
import requests
import pandas as pd
import torch
import re
import calendar
import streamlit as st
import zipfile
import gdown
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# --------------------------------------------------------
# CONFIGURATION: Google Drive file IDs for model/tokenizer
# --------------------------------------------------------
MODEL_ZIP_ID = "1bo5akUEzOYdz3OWnw37fleJhSvu6Y_FE"       # saved_model.zip
TOKENIZER_ZIP_ID = "1KxG4prB5Y5s8HOy11wPI37s7tAXU8xyB"   # saved_tokenizer.zip

MODEL_DIR = "saved_model"
TOKENIZER_DIR = "saved_tokenizer"


@st.cache_resource
def load_model_and_tokenizer():
    MODEL_DIR = "saved_model"
    TOKENIZER_DIR = "saved_tokenizer"

    # Google Drive file IDs (not full URLs)
    MODEL_ZIP_ID = "1bo5akUEzOYdz3OWnw37fleJhSvu6Y_FE"
    TOKENIZER_ZIP_ID = "1KxG4prB5Y5s8HOy11wPI37s7tAXU8xyB"

    if not os.path.exists(MODEL_DIR):
        gdown.download(id=MODEL_ZIP_ID, output="model.zip", quiet=False)
        with zipfile.ZipFile("model.zip", 'r') as zip_ref:
            zip_ref.extractall(MODEL_DIR)

    if not os.path.exists(TOKENIZER_DIR):
        gdown.download(id=TOKENIZER_ZIP_ID, output="tokenizer.zip", quiet=False)
        with zipfile.ZipFile("tokenizer.zip", 'r') as zip_ref:
            zip_ref.extractall(TOKENIZER_DIR)

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
