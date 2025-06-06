# Gmail vs Calendar Query Classifier using BERT (Streamlit + HF version)
# ----------------------------------------------------------------------

import os
import re
import calendar
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ----------------------
# Load model & tokenizer from Hugging Face
# ----------------------
@st.cache_resource
def load_model_and_tokenizer():
    repo = "manoj1222/gmail-calendar-model"
    tokenizer = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForSequenceClassification.from_pretrained(repo)
    model.eval()
    return tokenizer, model

# ----------------------
# Optional: Extract time range from text if month+year detected
# ----------------------
def extract_date_range(query):
    match = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})", query, re.IGNORECASE)
    if match:
        month = match.group(1).capitalize()
        year = int(match.group(2))
        month_num = list(calendar.month_name).index(month)
        num_days = calendar.monthrange(year, month_num)[1]
        return f"from: {month} 1, {year} to: {month} {num_days}, {year}"
    return None

# ----------------------
# Streamlit App UI
# ----------------------
st.title("Gmail vs Calendar Query Classifier")
st.header("Try")
st.markdown("Find emails with large attachments")
st.markdown("When is my next meeting with the design team?")
st.markdown("“Find my meetings for June 2025")


tokenizer, model = load_model_and_tokenizer()

user_input = st.text_input("Enter your query here")

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
