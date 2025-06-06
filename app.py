# Gmail vs Calendar Query Classifier using BERT (Streamlit Deployment)
# ---------------------------------------------------------------------
# This script loads a trained Roberta model and tokenizer to classify
# user queries into Gmail or Calendar using a Streamlit interface.

import os
import torch
import re
import calendar
import zipfile
import gdown
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Disable TensorFlow usage in HuggingFace Transformers
os.environ["USE_TF"] = "0"

# Google Drive File IDs for downloading model/tokenizer
MODEL_ZIP_ID = "1bo5akUEzOYdz3OWnw37fleJhSvu6Y_FE"       # saved_model.zip
TOKENIZER_ZIP_ID = "1KxG4prB5Y5s8HOy11wPI37s7tAXU8xyB"   # saved_tokenizer.zip

# Local paths
MODEL_DIR = "saved_model"
TOKENIZER_DIR = "saved_tokenizer"

# -----------------------------
# Function to download and unzip from Google Drive
# -----------------------------
def download_and_extract_from_gdrive(file_id, output_zip, extract_dir):
    if not os.path.exists(extract_dir):
        gdown.download(id=file_id, output=output_zip, quiet=False)
        with zipfile.ZipFile(output_zip, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

# -----------------------------
# Load model and tokenizer with caching
# -----------------------------
@st.cache_resource
def load_model_and_tokenizer():
    download_and_extract_from_gdrive(MODEL_ZIP_ID, "model.zip", MODEL_DIR)
    download_and_extract_from_gdrive(TOKENIZER_ZIP_ID, "tokenizer.zip", TOKENIZER_DIR)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model

# -----------------------------
# Function to extract date range from text
# -----------------------------
def extract_date_range(query):
    match = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})", query, re.IGNORECASE)
    if match:
        month = match.group(1).capitalize()
        year = int(match.group(2))
        month_num = list(calendar.month_name).index(month)
        num_days = calendar.monthrange(year, month_num)[1]
        return f"from: {month} 1, {year} to: {month} {num_days}, {year}"
    return None

# -----------------------------
# Streamlit Interface
# -----------------------------
st.title("📧 Gmail vs 📅 Calendar Query Classifier")

st.markdown("Enter a query related to Gmail or Calendar and let the model classify it!")

# Load model/tokenizer once
tokenizer, model = load_model_and_tokenizer()

# User input
user_input = st.text_input("🔍 Enter your query:")

if user_input:
    inputs = tokenizer(user_input, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    category = "Gmail" if prediction == 0 else "Calendar"
    st.success(f"🧠 Predicted Category: **{category}**")

    if category == "Calendar":
        time_range = extract_date_range(user_input)
        if time_range:
            st.info(f"📅 Extracted Time Range: {time_range}")
