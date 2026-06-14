# Phishing Email Detection Model

A machine learning-based phishing email detection system built using Python and Scikit-learn. 
This project analyzes email content, extracts important text features, and classifies emails as either **Phishing** or **Safe**.

## Features

- Train a machine learning model on phishing and legitimate emails
- Text feature extraction using TF-IDF Vectorization
- Classification using Random Forest Algorithm
- Detects suspicious keywords and URLs in emails
- Evaluates model performance using:
  - Accuracy Score
  - Classification Report
  - Confusion Matrix
- Tests the model on new unseen emails

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Machine Learning Workflow

1. Load and prepare email dataset
2. Split data into training and testing sets
3. Convert email text into numerical features using TF-IDF
4. Train Random Forest classifier
5. Evaluate model performance
6. Predict whether a new email is phishing or safe

## Project Goal

The goal of this project is to demonstrate how machine learning can be applied in cybersecurity to identify phishing attempts and improve email security.

## How to Run

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
