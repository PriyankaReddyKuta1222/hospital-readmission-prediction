
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Generate synthetic data
np.random.seed(42)

# Features
n_samples = 100000
ages = np.random.randint(18, 90, n_samples)
genders = np.random.choice(['Male', 'Female'], n_samples)
comorbidities = np.random.randint(0, 6, n_samples)
length_of_stay = np.random.randint(1, 15, n_samples)
medication_count = np.random.randint(1, 15, n_samples)
insurance_types = np.random.choice(['Private', 'Medicare', 'Medicaid', 'None'], n_samples)
admission_types = np.random.choice(['Emergency', 'Planned', 'Urgent'], n_samples)
diagnosis_codes = np.random.choice(['A00', 'B01', 'C02', 'D03', 'E04', 'F05', 'G06', 'H07', 'I08', 'J09'], n_samples)
readmission_flags = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])

# Create DataFrame
data = pd.DataFrame({
    'age': ages,
    'gender': genders,
    'comorbidities': comorbidities,
    'length_of_stay': length_of_stay,
    'medication_count': medication_count,
    'insurance_type': insurance_types,
    'admission_type': admission_types,
    'diagnosis_code': diagnosis_codes,
    'readmission_flag': readmission_flags
})

# Preprocess the data
data = pd.get_dummies(data, drop_first=True)  # One-hot encoding for categorical variables
X = data.drop("readmission_flag", axis=1)
y = data["readmission_flag"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_report_ = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Save the model
joblib.dump(model, 'logistic_model.pkl')

# Save evaluation metrics
evaluation_metrics = {
    'accuracy': accuracy,
    'classification_report': classification_report_,
    'confusion_matrix': conf_matrix.tolist()
}
with open('evaluation_metrics.json', 'w') as f:
    import json
    json.dump(evaluation_metrics, f)

# Plot feature importance
coef = model.coef_[0]
features = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=coef, y=features)
plt.title('Feature Importance')
plt.xlabel('Coefficient Value')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('feature_importance_plot.png')

# Show some results
print(f"Accuracy: {accuracy}")
print(classification_report_)
