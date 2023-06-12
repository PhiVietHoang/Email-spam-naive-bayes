import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.naive_bayes import MultinomialNB

# Load the dataset
data = pd.read_csv('mail_data.csv')

# Split the data into features and labels
X = data['Message']
y = data['Category']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the text data into numerical features using CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Predict the labels for the test set
y_pred = model.predict(X_test_vectorized.toarray())

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Generate and print the confusion matrix
confusion = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(confusion)

# Extract TP, FP, TN, FN values from the confusion matrix
tn, fp, fn, tp = confusion.ravel()
print('True Positives (TP):', tp)
print('False Positives (FP):', fp)
print('True Negatives (TN):', tn)
print('False Negatives (FN):', fn)

# Print classification report
report = classification_report(y_test, y_pred)
print('Classification Report:')
print(report)

# Calculate probabilities for the positive class
y_prob = model.predict_proba(X_test_vectorized.toarray())[:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test.map({'spam': 1, 'ham': 0}), y_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


with open('test.txt', 'r') as file:
    input_mail = [file.read()]

# Convert text to feature vectors
input_data_features = vectorizer.transform(input_mail)

# Making prediction
prediction = model.predict(input_data_features)
print(prediction)
