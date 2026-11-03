# FraudLens AI
### Intelligent Transaction Fraud Detection with GenAI Explanations
## 1. Project Overview

This project is an educational fraud detection prototype built using Python, Machine Learning, Streamlit, and Generative AI. It predicts whether a transaction may be fraudulent and generates a simple explanation of the model's prediction using a locally running language model through Ollama.

**Dataset limitation:** The project uses a public credit-card fraud dataset, not a debit-card-specific dataset. Therefore, it is a prototype demonstrating fraud detection techniques rather than a validated debit card fraud system.

## 2. Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Joblib
- Streamlit
- Ollama with Gemma 2B
- Requests

## 3. Main Features

- Displays transaction dataset statistics.
- Predicts the likelihood of fraud for a selected transaction.
- Displays the predicted class and an illustrative risk category.
- Generates a natural-language explanation using a local GenAI model.
- Shows the dataset's actual label after prediction for educational comparison.
- Displays the class distribution and sample transactions.

## 4. Project Structure

```text
FraudDetectionProject/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── creditcard.csv
├── models/
│   └── fraud_model.joblib
├── src/
│   ├── train_model.py
│   ├── predict.py
│   └── genai_explainer.py
└── screenshots/
```

## 5. How It Works

1. The dataset is loaded and prepared for model training.
2. The data is split into training and test sets using stratified sampling.
3. A Logistic Regression model with feature scaling and class weighting is trained.
4. The trained model is saved using Joblib.
5. The prediction module estimates a transaction's fraud probability.
6. Ollama's Gemma 2B model generates a human-readable explanation.
7. The Streamlit dashboard presents the prediction and dataset label for comparison.

## 6. Dataset Information

- Total transactions: 284,807
- Transactions labelled as fraud: 492
- Transactions labelled as not fraud: 284,315
- Training split: 80%
- Test split: 20%

The dataset is highly imbalanced because fraudulent transactions form only a small fraction of the data.

## 7. Model Evaluation

Results from the current held-out test set:

| Metric | Fraud-class result |
|---|---:|
| Precision | 0.06 |
| Recall | 0.92 |
| F1-score | 0.11 |
| Average precision | 0.719 |

Confusion matrix:

| Actual / Predicted | Not flagged | Potential fraud |
|---|---:|---:|
| Not fraud | 55,478 | 1,386 |
| Fraud | 8 | 90 |

The model detects many fraudulent cases, but it also generates many false alarms. Precision and the balance between precision and recall need improvement before any real-world application.

## 8. Running the Project

### Install dependencies

Activate the project's virtual environment, then run:

```powershell
pip install -r requirements.txt
```

### Train the model

```powershell
python src/train_model.py
```

### Start Ollama

Ensure Ollama is installed and the Gemma 2B model is available. The application expects Ollama's local API at `http://localhost:11434`.

```powershell
ollama run gemma:2b
```

Keep Ollama running. If the command is not recognized in PowerShell, use the Ollama executable's installed path or add its folder to PATH.

### Run the dashboard

From the project root:

```powershell
python -m streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

## 9. Limitations and Future Improvements

- The benchmark contains credit-card transactions rather than debit-card-specific transactions.
- A predicted probability is a model estimate, not proof of fraud.
- The illustrative risk categories have not been validated for banking use.
- The current model has low fraud precision and generates false positives.
- Future work could compare additional models, tune the decision threshold, improve precision and recall, and evaluate models using precision-recall curves.
- Real financial deployment would require appropriate data, rigorous validation, security controls, privacy safeguards, and expert review.

## 10. Disclaimer

This project is intended for educational and demonstration purposes only. It must not be used to approve, decline, or investigate real financial transactions without extensive additional validation.