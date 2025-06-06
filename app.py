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

# Load saved model and tokenizer from training phase
@st.cache_resource
def load_best_model():
    tokenizer = AutoTokenizer.from_pretrained("saved_tokenizer")
    model = AutoModelForSequenceClassification.from_pretrained("saved_model")  # correct usage
    model.eval()
    return tokenizer, model

tokenizer, model = load_best_model()


# Extract time range if query contains month-year
# Example: "meetings for June 2025" → from: June 1, 2025 to: June 30, 2025

def extract_date_range(query):
    match = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)\\s+(\\d{4})", query, re.IGNORECASE)
    if match:
        month = match.group(1).capitalize()
        year = int(match.group(2))
        month_num = list(calendar.month_name).index(month)
        num_days = calendar.monthrange(year, month_num)[1]
        return f"from: {month} 1, {year} to: {month} {num_days}, {year}"
    return None

# Classify input query into Gmail or Calendar

def classify_query(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    category = "Gmail" if prediction == 0 else "Calendar"
    time_range = extract_date_range(text) if category == "Calendar" else None
    return category, time_range

# Streamlit Interface to input query and display result
st.title("📬 Gmail vs Calendar Query Classifier")

user_input = st.text_input("Enter your query below:")
if user_input:
    prediction, range_info = classify_query(user_input)
    st.success(f"Predicted Category: {prediction}")
    if range_info:
        st.info(f"📅 Time Range Extracted: {range_info}")
