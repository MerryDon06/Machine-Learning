import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
data = pd.read_csv('student_sleep.csv')

# Exploratory Data Analysis (EDA)
# Univariate Analysis
for column in data.select_dtypes(include=['float64', 'int64']).columns:
    plt.figure()
    sns.histplot(data[column], kde=True)
    plt.title(f'Distribution of {column}')

# Bivariate Analysis
sns.pairplot(data, hue='Physical_Activity')  # Replace 'target_column' with the actual target column name

# Multivariate Analysis (Heatmap)
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')

# Preprocessing
# Fix for FutureWarning: Fill missing values only for numeric columns
numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())

# Encode categorical variables
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Split data into features and target
X = data.drop(columns='Sleep_Duration')  # Replace 'target_column' with the actual target column name
y = data['Sleep_Quality']  # Replace 'target_column' with the actual target column name

# Standardization for GaussianNB only
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Ensure non-negative values for MultinomialNB
X_non_negative = np.maximum(0, X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train_mnb, X_test_mnb, _, _ = train_test_split(X_non_negative, y, test_size=0.2, random_state=42)

# Train Gaussian Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)

# Train Multinomial Naive Bayes
mnb = MultinomialNB()
mnb.fit(X_train_mnb, y_train)
y_pred_mnb = mnb.predict(X_test_mnb)

# Evaluation
# Accuracy Scores
accuracy_gnb = accuracy_score(y_test, y_pred_gnb)
accuracy_mnb = accuracy_score(y_test, y_pred_mnb)
print(f"Accuracy of GaussianNB: {accuracy_gnb}")
print(f"Accuracy of MultinomialNB: {accuracy_mnb}")

# Confusion Matrices
cm_gnb = confusion_matrix(y_test, y_pred_gnb)
cm_mnb = confusion_matrix(y_test, y_pred_mnb)

sns.heatmap(cm_gnb, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - GaussianNB')
plt.show()

sns.heatmap(cm_mnb, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - MultinomialNB')
plt.show()

# Cross-Validation
cv_scores_gnb = cross_val_score(gnb, X_scaled, y, cv=5)
cv_scores_mnb = cross_val_score(mnb, X_non_negative, y, cv=5)
print(f"Cross-Validation Scores (GaussianNB): {cv_scores_gnb}")
print(f"Cross-Validation Scores (MultinomialNB): {cv_scores_mnb}")

# ROC Curve
fpr_gnb, tpr_gnb, _ = roc_curve(y_test, gnb.predict_proba(X_test)[:, 1])
roc_auc_gnb = auc(fpr_gnb, tpr_gnb)

fpr_mnb, tpr_mnb, _ = roc_curve(y_test, mnb.predict_proba(X_test_mnb)[:, 1])
roc_auc_mnb = auc(fpr_mnb, tpr_mnb)

plt.figure()
plt.plot(fpr_gnb, tpr_gnb, label=f'GaussianNB (AUC = {roc_auc_gnb:.2f})')
plt.plot(fpr_mnb, tpr_mnb, label=f'MultinomialNB (AUC = {roc_auc_mnb:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='best')
plt.show()

# Interpretation
print("Classification Report - GaussianNB")
print(classification_report(y_test, y_pred_gnb))

print("Classification Report - MultinomialNB")
print(classification_report(y_test, y_pred_mnb))
