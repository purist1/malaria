# Malaria Occurrence Prediction System

An advanced clinical decision support tool that uses supervised machine learning to predict binary malaria occurrence (`Malaria Positive` / `Malaria Negative`) based on patient demographics, clinical length of stay, and 11 distinct clinical symptoms. This project is specifically tailored for deployment and evaluation in healthcare facilities across Kogi State, Nigeria.

---

## 1. Thesis Report Reference & Background

This section provides all details required to compile the introduction, methodology, results, and discussion sections of an academic thesis or technical report.

### Project Summary
* **Domain:** Healthcare Informatics / Epidemiological Predictive Modeling / Applied Machine Learning
* **Problem Statement:** Early and accurate diagnosis of malaria is crucial in endemic areas like Sub-Saharan Africa. Clinical symptoms often overlap with other febrile illnesses, leading to misdiagnoses or delayed treatment. This system builds predictive classifiers using clinical symptoms and basic demographic indicators to assess patient-level malaria risk, acting as a triage/screening assistant.
* **Target Outcome:** A binary prediction:
  * `0`: Malaria Negative / Mild (non-severe/unconfirmed)
  * `1`: Malaria Positive / Severe (confirmed case)

---

## 2. Dataset & Feature Reference

The models are trained and validated on `dataset/Malaria_Dataset.csv`.

### Cohort Characteristics & Feature List
The dataset comprises clinical records of patients presenting with febrile symptoms. The features are categorized as follows:

1. **Demographic & Geographic Features:**
   * **Age** (Continuous, scaled): Patient's age in years.
   * **Sex** (Categorical, One-Hot encoded): `Male` or `Female`.
   * **Residence Area** (Categorical, One-Hot encoded): Patient's residential area within Kogi State, Nigeria. Options include:
     * `Lokoja` (State Capital / Urban)
     * `Idah` (Eastern Zone / Riverine)
     * `Anyigba` (Eastern Zone / University Community)
     * `Kabba` (Western Zone / Semi-urban)
     * `Okene` (Central Zone / Industrial)

2. **Temporal Feature (Engineered):**
   * **Length of Stay (days)** (`length_of_stay`): Computed as `Discharge_Date - Date of Admission (DOA)` in integer days. Captures the clinical duration before discharge, which serves as a proxy for disease severity and resource utilization.

3. **Clinical Symptom Flags** (Binary, `0 = No`, `1 = Yes`):
   * **Fever** (Core systemic indicator)
   * **Headache**
   * **Abdominal Pain**
   * **General Body Malaise** (Fatigue/body weakness)
   * **Dizziness**
   * **Vomiting**
   * **Confusion** (Indicates potential neurological involvement / cerebral malaria risk)
   * **Backache**
   * **Chest Pain**
   * **Coughing**
   * **Joint Pain**

### Administrative & Leakage Columns Dropped
To prevent data leakage and ensure the model only relies on features available at the initial clinical presentation, the following columns were excluded from training:
* `IP_Number`: Unique patient identifier (no predictive capability).
* `Primary_Code` / `Diagnosis_Type`: International Classification of Diseases (ICD) codes and post-hoc clinical descriptions entered *after* final laboratory confirmation.
* `DOA` / `Discharge_Date`: Extracted into `length_of_stay` and removed to prevent raw date leakage.
* `Risk_Score`: A post-hoc administrative score derived from the symptoms themselves, which would introduce artificial collinearity.

---

## 3. Data Preprocessing & Pipeline Architecture

The preprocessing pipeline ensures data cleanliness and prevents leakages. It follows a strict workflow:

1. **Header Normalization:** Converts column headers to standard lowercase `snake_case`.
2. **Missing Value Imputation:**
   * Computed on the **training fold only** to prevent leakage into the test fold.
   * Continuous features (`age`, `length_of_stay`) are imputed using the **median**.
   * Categorical features are imputed using the **mode**.
3. **Deduplication:** Duplicate records are automatically removed.
4. **Encoding:** Categorical features are transformed using `OneHotEncoder`.
5. **Feature Scaling:** Numerical features and binary symptom flags are standardized using `StandardScaler` to ensure zero mean and unit variance, which is critical for distance-based estimators like SVM and Logistic Regression.
6. **Train/Test Splitting:** An 80/20 stratified split is used to preserve the class label distribution between training and test sets.
7. **Class Imbalance Mitigation:** All classifiers are trained with the `class_weight='balanced'` parameter to handle differences in sample distribution.

---

## 4. Model Architectures & Configurations

Four supervised learning classifiers are built and optimized:

| Classifier | Hyperparameters / Setup | Rationale |
|---|---|---|
| **Logistic Regression** | `max_iter: 1000`, `C: 1.0`, `class_weight: "balanced"` | Establishes a linear baseline; easy to interpret. |
| **Decision Tree** | `max_depth: 8`, `min_samples_leaf: 5`, `class_weight: "balanced"` | Captures non-linear decision boundaries and rule-based structures. |
| **Random Forest** | `n_estimators: 200`, `max_depth: 12`, `min_samples_leaf: 3`, `class_weight: "balanced"` | Ensemble bagging model; minimizes overfitting while providing robust feature importance. |
| **Support Vector Machine (SVM)** | `kernel: "rbf"`, `C: 1.0`, `probability: True`, `class_weight: "balanced"` | Captures complex, high-dimensional non-linear interactions using the Radial Basis Function kernel. |

---

## 5. Experimental Results & Performance Comparison

The models were trained and evaluated on the test dataset.

### Evaluation Metrics Summary
The model performance scores are as follows:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Support Vector Machine (SVM)** | **95.08%** | 95.42% | **97.86%** | **96.62%** | **98.37%** |
| **Random Forest** | 93.54% | **98.19%** | 92.74% | 95.38% | 98.27% |
| **Logistic Regression** | 93.23% | 96.09% | 94.44% | 95.26% | 97.60% |
| **Decision Tree** | 91.69% | 96.41% | 91.88% | 94.09% | 93.99% |

### Key Findings & Thesis Insights
* **Best Model Selection:** **Support Vector Machine (SVM)** with the RBF kernel achieved the highest performance across almost all metrics, boasting an **F1-Score of 96.62%** and a **ROC-AUC of 98.37%**. Its extremely high **Recall (97.86%)** is particularly desirable in a clinical screening tool to minimize false negatives (missed cases of malaria).
* **Random Forest** achieved the highest **Precision (98.19%)**, making it the most conservative model for predicting positive cases (lowest false-positive rate).

---

## 6. Model Explainability & Feature Importance

Feature importance was evaluated using the Mean Gini Importance (feature importance score) extracted from the Random Forest model.

### Feature Importance Ranking
Below is the ranking of patient features based on their contribution to the prediction:

1. **Fever** (22.32%) - The primary systemic predictor of malaria occurrence.
2. **General Body Malaise** (18.39%) - Highly correlated with systemic infection.
3. **Age** (12.13%) - Demographics play a key role; younger patients and elderly cohorts display different clinical vulnerability profiles.
4. **Vomiting** (9.27%) - Gastrointestinal distress is a strong marker.
5. **Headache** (7.73%)
6. **Dizziness** (7.69%)
7. **Abdominal Pain** (5.52%)
8. **Length of Stay** (3.80%) - Indicates clinical course complexity.
9. **Confusion** (2.75%) - Neurological symptom indicating severe progression.
10. **Coughing** (1.26%)
11. **Chest Pain** (1.20%)
12. **Backache** (1.10%)
13. **Joint Pain** (1.08%)
14. **Sex (Male/Female)** (~1.84% combined)
15. **Residence Area (Kogi Locations)** (~3.93% combined: Idah: 0.89%, Anyigba: 0.81%, Lokoja: 0.79%, Okene: 0.77%, Kabba: 0.67%)

---

## 7. Streamlit Dashboard & Clinical Workflow

The Streamlit dashboard (`dashboard/app.py`) provides an interactive interface for clinical decision-making.

### Key Features
1. **Interactive Form:** Allows inputting demographics (Sex, Age, Residence Area), length of stay, and 11 symptoms to predict malaria risk.
2. **Model Selector:** Dropdown menu to switch between the models (SVM, Random Forest, Logistic Regression, Decision Tree) or auto-select the best performing one.
3. **Decision Threshold Slider:** Allows adjustment of the probability threshold (default `0.5`).
   * *Clinical utility:* Lowering the threshold (e.g., to `0.35`) increases the model's sensitivity (recall), which is useful in screening scenarios to catch all possible cases.
4. **Retrain Interface:** Sidebar button to retrain all models on the fly when new patient records are appended to the raw dataset.
5. **Performance Visualizations:** Includes ROC curves, comparison charts, and confusion matrices.
6. **Data Explorer:** Displays raw and processed datasets and quality metrics.

---

## 8. Ethical, Regulatory & Data Compliance

* **De-identification:** All patient records were completely anonymized (such as stripping patient IDs and names).
* **Clinical Disclaimer:** This tool serves as clinical decision support only and does not replace diagnostic laboratory tests (blood smear microscopy, Rapid Diagnostic Tests).
* **Data Privacy:** Compliant with the National Information Technology Development Agency (NITDA) Nigeria Data Protection Regulation (NDPR) and WHO health data guidelines.

---

## 9. Developer Guide & Execution Commands

### Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run Model Training Pipeline
```bash
python scripts/train_model.py --config config/config.yaml
```

### Run Single Sample CLI Prediction
```bash
python scripts/predict.py --sex Male --age 28 --residence-area Lokoja --fever Yes --headache Yes
```

### Run Streamlit App
```bash
streamlit run dashboard/app.py
```
