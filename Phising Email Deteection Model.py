import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# 1. Load Data (Using Mock Data for Testing)
# ==========================================
# In a real scenario, replace this with: 
# df = pd.read_csv('your_dataset.csv')
data = {
    'email_text': [
        "URGENT: Your account has been compromised. Click http://secure-login-update.com to verify your password.",
        "Hey team, just a reminder about our meeting at 3 PM tomorrow. See you then!",
        "Claim your $1000 Walmart gift card now! Limited time offer. Visit http://free-money-now.net.",
        "Attached is the Q3 financial report for your review. Let me know if you have questions.",
        "Security Alert: We detected unusual login activity. Please reset your credentials immediately.",
        "Can we reschedule our lunch to Friday? I have a conflict on Thursday.",
        "Your invoice #9932 is overdue. Please pay immediately to avoid account suspension: http://fake-billing.com",
        "Happy birthday! Hope you have a fantastic day celebrating with family."
    ],
    'label': ['Phishing', 'Safe', 'Phishing', 'Safe', 'Phishing', 'Safe', 'Phishing', 'Safe']
}

df = pd.DataFrame(data)

# ==========================================
# 2. Split Data into Training and Testing Sets
# ==========================================
X = df['email_text']
y = df['label']

# Standard 80/20 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# ==========================================
# 3. Build the Machine Learning Pipeline
# ==========================================
# A pipeline ensures text vectorization and classification happen seamlessly
model_pipeline = Pipeline([
    # Step A: Feature Extraction
    # TfidfVectorizer converts raw text into a matrix of TF-IDF features.
    # It automatically handles keywords, URLs, and removes standard stop words.
    ('vectorizer', TfidfVectorizer(stop_words='english', max_features=5000)),
    
    # Step B: Classification
    # Random Forest is highly effective for text classification and resistant to overfitting.
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# ==========================================
# 4. Train the Model
# ==========================================
print("Training the model...")
model_pipeline.fit(X_train, y_train)

# ==========================================
# 5. Evaluate the Model
# ==========================================
print("Evaluating the model...\n")
predictions = model_pipeline.predict(X_test)

# Calculate Accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")

# Detailed Classification Report (Precision, Recall, F1-Score)
print("Classification Report:")
print(classification_report(y_test, predictions))

# ==========================================
# 6. Display the Confusion Matrix
# ==========================================
# Generating a professional visual heatmap for the confusion matrix
cm = confusion_matrix(y_test, predictions, labels=['Safe', 'Phishing'])

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Predicted Safe', 'Predicted Phishing'],
            yticklabels=['Actual Safe', 'Actual Phishing'])

plt.title('Phishing Detection Confusion Matrix', pad=15)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# ==========================================
# 7. Test with a New, Unseen Email
# ==========================================
new_email = ["Verify your bank details here: http://chase-update-info.com"]
prediction = model_pipeline.predict(new_email)
print(f"\nTest Email: '{new_email[0]}'")
print(f"Prediction: {prediction[0]}")