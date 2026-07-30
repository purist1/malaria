# Malaria Occurrence Prediction System

An advanced clinical decision support tool that uses supervised machine learning to predict binary malaria occurrence (`Malaria Positive` / `Malaria Negative`) based on patient demographics, environmental context, preventive measures, and clinical symptoms. This project has been expanded to support generalizable locality across **Sub-Saharan Africa**.

---

## 1. Thesis Report Reference & Background

This section provides all details required to compile the introduction, methodology, results, and discussion sections of an academic thesis or technical report.

### Project Summary
* **Domain:** Healthcare Informatics / Epidemiological Predictive Modeling / Applied Machine Learning
* **Problem Statement:** Early and accurate diagnosis of malaria is crucial in endemic areas across Sub-Saharan Africa. Clinical symptoms often overlap with other febrile illnesses, leading to misdiagnoses or delayed treatment. This system builds predictive classifiers using clinical presentation, environmental indicators, and preventive behavior to assess patient-level malaria risk, acting as a triage/screening assistant.
* **Target Outcome:** A binary prediction:
  * `0`: Malaria Negative
  * `1`: Malaria Positive

---

## 2. Dataset & Feature Reference

The models are trained and validated on the Africa-wide malaria dataset (`electricsheepafrica/africa-synth-malaria-malaria-dataset-all`), cached locally at `dataset/africa_malaria_hf/train.csv`.

### Cohort Characteristics & Feature List
The dataset comprises clinical records of patients presenting with febrile symptoms across Sub-Saharan Africa. The features are categorized as follows:

1. **Demographic & Geographic Features:**
   * **Age** (`age_years`, Continuous, scaled): Patient's age in years.
   * **Sex** (`sex`, Categorical, One-Hot encoded): `Male` or `Female`.
   * **Residence Setting** (`residence`, Categorical, One-Hot encoded): General location classification (`Urban` or `Rural`). Replaces hardcoded region-specific LGAs to enable broad generalizability.

2. **Environmental & Preventive Features:**
   * **Transmission Season** (`season`, Categorical, One-Hot encoded): Climate season (`Rainy` or `Dry`).
   * **Mosquito Net Usage** (`uses_mosquito_net`, Binary flag `0/1`): Regular use of insecticide-treated bed nets.

3. **Clinical & Laboratory Indicators:**
   * **Fever Duration (days)** (`fever_days`, Integer, scaled): Duration of consecutive fever symptoms.
   * **Hemoglobin Level** (`hemoglobin_g_dl`, Continuous, scaled, optional): Blood hemoglobin level in g/dL. Imputed with training-set median if unmeasured at triage.

4. **Clinical Symptom Flags** (Binary, `0 = No`, `1 = Yes`):
   * **Fever** (`has_fever`)
   * **Chills / Rigors** (`has_chills`)
   * **Headache** (`has_headache`)
   * **Vomiting / Nausea** (`has_vomiting`)
   * **Diarrhea** (`has_diarrhea`)
   * **General Body Weakness** (`has_weakness`)

### Administrative & Leakage Columns Dropped
To prevent data leakage and ensure the model only relies on features available at initial triage, the following post-hoc columns are excluded from training:
* `patient_id`: Unique identifier.
* `age_months`, `age_group`: Redundant age representations.
* `parasitemia_level`, `parasitemia_count`, `plasmodium_species`: Post-diagnosis microscopic lab findings.
* `anemia_status`: Derived post-hoc from Hb.
* `outcome`, `malaria_probability_score`: Target leakage variables.
* `severe_malaria`, `cerebral_malaria`, `respiratory_distress`, `shock`, `acute_kidney_injury`: Clinical severity flags assigned after diagnosis confirmation.

---

## 3. Data Preprocessing & Pipeline Architecture

The preprocessing pipeline ensures data cleanliness and prevents leakage. It follows a strict workflow:

1. **Header Normalization:** Converts column headers to standard lowercase `snake_case`.
2. **Boolean Casting:** Transforms boolean flags (`True`/`False`) into integer values (`1`/`0`).
3. **Missing Value Imputation:**
   * Computed on the **training fold only**.
   * Continuous features (`age_years`, `fever_days`, `hemoglobin_g_dl`) are imputed using the **median**.
   * Categorical features (`sex`, `residence`, `season`) are imputed using the **mode**.
4. **Deduplication:** Duplicate records are automatically removed.
5. **Encoding:** Categorical features (`sex`, `residence`, `season`) are transformed using `OneHotEncoder`.
6. **Feature Scaling:** Continuous features and binary symptom flags are standardized using `StandardScaler`.
7. **Train/Test Splitting:** An 80/20 stratified split preserves class label distribution between training and test sets.
8. **Class Imbalance Handling:** Classifiers are trained with `class_weight='balanced'`.

---

## 4. Model Architectures & Configurations

Four supervised learning classifiers are built and optimized:

| Classifier | Hyperparameters / Setup | Rationale |
|---|---|---|
| **Logistic Regression** | `max_iter: 1000`, `C: 1.0`, `class_weight: "balanced"` | Establishes a linear baseline; highly interpretable. |
| **Decision Tree** | `max_depth: 8`, `min_samples_leaf: 5`, `class_weight: "balanced"` | Captures non-linear decision boundaries and rule structures. |
| **Random Forest** | `n_estimators: 200`, `max_depth: 12`, `min_samples_leaf: 3`, `class_weight: "balanced"` | Ensemble bagging model; minimizes overfitting. |
| **Support Vector Machine (SVM)** | `kernel: "rbf"`, `C: 1.0`, `probability: True`, `class_weight: "balanced"` | Captures high-dimensional non-linear interactions using RBF kernel. |

---

## 5. Streamlit Dashboard & Clinical Workflow

The Streamlit dashboard (`dashboard/app.py`) provides an interactive interface for clinical decision-making.

### Key Features
1. **Interactive Form:** Allows inputting patient demographics (`Sex`, `Residence: Urban/Rural`), environmental context (`Season: Rainy/Dry`), preventive net usage, fever duration, optional hemoglobin lab value, and 6 core clinical symptoms.
2. **Model Selector:** Dropdown menu to switch between models (SVM, Random Forest, Logistic Regression, Decision Tree) or auto-select the best performing classifier.
3. **Decision Threshold Slider:** Allows adjustment of probability threshold (default `0.5`).
4. **Retrain Interface:** Sidebar button to retrain all models on the fly.
5. **Performance Visualizations:** Displays ROC curves, comparison bar charts, and confusion matrices.
6. **Data Explorer:** Displays raw and processed datasets and quality metrics.

---

## 6. Ethical, Regulatory & Data Compliance

* **De-identification:** All records are fully anonymized.
* **Clinical Disclaimer:** This tool serves as clinical decision support only and does not replace diagnostic laboratory tests (microscopy, RDTs).
* **Data Privacy:** Compliant with NITDA Nigeria Data Protection Regulation (NDPR) and WHO health data guidelines.

---

## 7. Developer Guide & Execution Commands

### Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Required Dependencies:**
- numpy>=1.24.0
- pandas>=2.0.0
- scikit-learn>=1.3.0
- joblib>=1.3.0
- pyyaml>=6.0
- matplotlib>=3.7.0
- seaborn>=0.12.0
- streamlit>=1.28.0
- plotly>=5.18.0
- pytest>=7.0.0
- imbalanced-learn>=0.11.0
- nbformat>=5.9.0
- jupyter>=1.0.0
- datasets>=2.14.0 (for HuggingFace dataset download)

### Download Dataset
```bash
python scripts/download_dataset.py
```

### Run Model Training Pipeline
```bash
python scripts/train_model.py --config config/config.yaml
```

### Run Single Sample CLI Prediction
```bash
python scripts/predict.py --sex Male --residence Rural --season Rainy --age-years 25 --fever-days 3 --has-fever 1 --has-chills 1
```

### Run Tests
```bash
pytest tests/ -v
```

### Generate EDA Figures
```bash
python scripts/generate_eda.py --config config/config.yaml
```

### Run Streamlit App
```bash
streamlit run dashboard/app.py
```
