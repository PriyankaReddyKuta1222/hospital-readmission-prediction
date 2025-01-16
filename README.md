# hospital-readmission-prediction
Predictive model for hospital readmissions using logistic regression and healthcare analytics
# Predictive Model for Hospital Readmission

## Overview
This project develops a predictive model for hospital readmission using logistic regression. The model processes and analyzes synthetic healthcare data, focusing on identifying patterns and factors contributing to patient readmissions. Additionally, it includes interactive visualizations to communicate insights effectively.

---

## Dataset
The dataset used in this project is synthetic, consisting of 100,000 records with the following features:
- **age**: Patient's age
- **gender**: Patient's gender (Male/Female)
- **comorbidities**: Number of comorbid conditions
- **length_of_stay**: Hospital stay in days
- **medication_count**: Number of prescribed medications
- **insurance_type**: Type of insurance (Private, Medicaid, Medicare, None)
- **admission_type**: Type of hospital admission (Emergency, Planned, Urgent)
- **diagnosis_code**: ICD-10 diagnosis code
- **readmission_flag**: Target variable indicating if the patient was readmitted (1 = Yes, 0 = No)

---

## Project Files
1. **`logistic_model.pkl`**: Trained logistic regression model.
2. **`evaluation_metrics.json`**: Contains evaluation results including accuracy, classification report, and confusion matrix.
3. **`feature_importance_plot.png`**: Visual representation of feature importance.
4. **`cleaned_data.csv`**: Preprocessed dataset used for training and evaluation.

---

## Requirements
Install the necessary Python libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

---

## Usage

### 1. Load and Explore the Dataset
```python
import pandas as pd

data = pd.read_csv("cleaned_data.csv")
print(data.head())
```

### 2. Test the Trained Model
```python
import joblib

# Load the model
model = joblib.load("logistic_model.pkl")

# Load data
X = data.drop("readmission_flag", axis=1)
y = data["readmission_flag"]

# Make predictions
predictions = model.predict(X)
print(predictions[:10])
```

### 3. Visualize Feature Importance
Open `feature_importance_plot.png` to view the key factors influencing the model.

---

## Results
- **Accuracy**: Check `evaluation_metrics.json` for detailed performance metrics.
- **Confusion Matrix**: Provides insights into prediction success rates.

---

## Visualization
This project leverages Tableau or Power BI for creating dashboards. Exported datasets (e.g., `cleaned_data.csv`) can be imported into these tools for advanced visualizations.

---

## Repository Structure
```plaintext
.
├── logistic_model.pkl               # Trained model file
├── evaluation_metrics.json          # Evaluation metrics
├── feature_importance_plot.png      # Feature importance visualization
├── cleaned_data.csv                 # Preprocessed dataset
├── README.md                        # Project documentation
```

---

## Future Work
- Integrate additional machine learning models for comparison.
- Analyze real-world datasets for enhanced model applicability.
- Deploy the model as a REST API for real-time predictions.

---

## Author
Priyanka Reddy Kuta

---

## License
This project is licensed under none.
