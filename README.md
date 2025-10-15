# robust-explainable-phishing-classification
A BERT-based phishing email classifier with adversarial robustness and explainability (LIME + LLM fallback).

 ## Features

DistilBERT phishing detector (fine-tuned on open datasets)

LIME token-level explanations

LLM-based natural language explanations

FGM adversarial training for robustness

Fully reproducible on Google Colab


# Setup (Local)

git clone https://github.com/saj-stack/robust-explainable-phishing-classification

cd robust-explainable-phishing-classification

pip install transformers datasets accelerate torch numpy pandas scikit-learn lime pyspellchecker sentence-transformers

python robust_explainable_phishing_mini.py


## Google Colab Setup

!pip install transformers datasets accelerate torch numpy pandas scikit-learn lime pyspellchecker sentence-transformers

!git clone https://github.com/saj-stack/robust-explainable-phishing-classification

%cd robust-explainable-phishing-classification

!python robust_explainable_phishing_mini.py


## Data and Code

Dataset: Subset of the publicly available Phishing Email Dataset on Hugging Face.

Code Repository: https://github.com/saj-stack/robust-explainable-phishing-classification
