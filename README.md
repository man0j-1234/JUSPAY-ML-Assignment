# JUSPAY-ML-Assignment
Gmail vs Calendar Query Classifier
Here’s a clean and professional `README.md` file for your **Gmail vs Calendar Query Classifier** project, tailored for GitHub. It highlights your use of Hugging Face, Streamlit, and includes clear setup and usage instructions:

---

```markdown
# 📧 Gmail vs 📅 Calendar Query Classifier using BERT

This project is a text classification system built using **BERT (RoBERTa-base)** to distinguish between Gmail-related and Calendar-related queries. It includes model training, evaluation using Stratified K-Fold validation, and an interactive **Streamlit** app deployed via Hugging Face Spaces or local execution.

## 🚀 Features

- Trained using Hugging Face Transformers
- Balanced dataset with real-world queries
- Optional date extraction from calendar queries
- Clean deployment via Hugging Face model hub + Streamlit interface

## 🔍 Problem Statement

Users often issue natural language queries to assistants. These queries may relate to Gmail (e.g., "Show me unread emails") or Calendar (e.g., "Add meeting at 3 PM"). The goal is to **automatically classify** each query into either `Gmail` or `Calendar`.

> The key challenge lies in the subtle linguistic overlap between the two types of queries.

## 📂 Project Structure

```

.
├── app.py                # Streamlit UI that loads model from Hugging Face
├── Text-Classifier.ipynb # Model training + evaluation + upload script
├── requirements.txt      # Required packages
└── README.md             # Project description

````

## 🤖 Model

- Base: `roberta-base`
- Training: Stratified K-Fold (3 folds)
- Loss: CrossEntropyLoss (optionally class-weighted)
- Metrics: Accuracy, F1-score, Confusion Matrix

The final trained model and tokenizer are pushed to Hugging Face:

👉 [View on Hugging Face](https://huggingface.co/manoj1222/gmail-calendar-model)

## 🖥️ Streamlit App

### ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
````

### 🌐 Or Try Online

You can launch the app directly using Streamlit Cloud or Hugging Face Spaces.

## 🧠 Example Queries

| Query                                  | Predicted Category |
| -------------------------------------- | ------------------ |
| "Find emails from Sarah"               | Gmail              |
| "Add meeting with team tomorrow at 5"  | Calendar           |
| "Check messages about project updates" | Gmail              |
| "What's on my schedule this Friday?"   | Calendar           |
| Find my meetings for June 2025 "       | Calendar ,Extracted Time Range: from: June 1, 2025 to: June 30, 2025 |

## 📈 Results

* **Balanced F1 Scores** for both classes
* Handles implicit language variations well
* Accurate time-range extraction from Calendar queries
  

## 🙋‍♂️ Author

👤 **Manoj**

Feel free to connect or give feedback!

## 📜 License

MIT License

```

---

Let me know if you also want a short summary version or a version tailored for **Hugging Face model card** (`README.md` on HF Hub).
```
