# Proposed System Architecture & Process Diagrams

This document contains Mermaid diagrams detailing the architecture, data flows, use cases, and logical entities of the Malaria Occurrence Prediction System.

---

## 1. System Pipeline Flowchart
This diagram outlines the progression from raw demographics and clinical symptoms inputs, through preprocessing and feature selection, model training and selection, up to inference and result visualization.

```mermaid
flowchart TD
    Inputs["1. Input Variables<br>(Demographics, Stay Duration, Symptoms)"]
    Preproc["2. Preprocessing & Feature Selection<br>(Normalization, Imputation, One-Hot Encoding, Scaling)"]
    Models["3. Model Training & Comparison<br>(Train LR, DT, RF, SVM; Choose best by F1-Score)"]
    Prediction["4. Inference & Prediction<br>(Single/Batch Prediction + Custom Threshold)"]
    Visualization["5. Result Visualization<br>(ROC Curves, Confusion Matrices, Performance Charts)"]

    Inputs --> Preproc --> Models --> Prediction --> Visualization
    Models -.-> Visualization
```

---

## 2. Use Case Diagram of the Proposed System
This diagram shows the actions that the Healthcare Professional and System Administrator can perform within the boundary of the prediction system.

```mermaid
flowchart LR
    Admin["System Administrator / Developer"]
    Clinician["Healthcare Professional<br>(Clinician / Nurse)"]

    subgraph System ["Malaria Occurrence Prediction System"]
        UC1(["Input Patient demographics & symptoms"])
        UC2(["Predict Malaria Occurrence & Risk"])
        UC3(["Adjust Classification Threshold"])
        UC4(["Retrain Predictive ML Models"])
        UC5(["View Comparative Model Performance"])
        UC6(["Preview Dataset & Statistics"])
    end

    Clinician --> UC1
    Clinician --> UC2
    Clinician --> UC3
    Clinician --> UC5
    Clinician --> UC6

    Admin --> UC4
    Admin --> UC5
    Admin --> UC6

    UC1 -.->|includes|-> UC2
```

---

## 3. Data Flow Diagram (Level 0) of the Proposed System
The Context Diagram (DFD Level 0) models the system as a single process, showing the interfaces between the prediction engine and external entities.

```mermaid
flowchart LR
    User["Healthcare Professional"]
    Admin["System Administrator"]
    DataStore[("Dataset Store<br>(Malaria_Dataset.csv)")]
    System["0.0<br>Malaria Occurrence Prediction System"]

    User -- "Patient Symptoms & Demographics" --> System
    System -- "Malaria Risk Predictions & Probability" --> User
    
    Admin -- "Retrain Command & Config settings" --> System
    System -- "Model Comparison Metrics & Plots" --> Admin

    DataStore -- "Historical Patient Records" --> System
    System -- "Updated Processed Dataset" --> DataStore
```

---

## 4. Sequence Diagram of the Prediction Process
This diagram illustrates the step-by-step messaging and invocation sequence when a healthcare professional requests a malaria risk prediction.

```mermaid
sequenceDiagram
    autonumber
    actor User as Healthcare Professional
    participant UI as Streamlit Web App (app.py)
    participant Pred as Inference Engine (predict.py)
    participant Disk as Artifact Store (models/)

    User->>UI: Enter Demographics & Symptoms (Fever, Malaise, etc.)
    User->>UI: Click "Predict Malaria Risk"
    UI->>Disk: Load metadata.joblib (defaults, scaler, encoder)
    Disk-->>UI: Return pipeline estimators & metadata
    UI->>Disk: Load best_model.joblib (or selected model)
    Disk-->>UI: Return trained model instance
    UI->>Pred: predict_single_record(record, model, metadata)
    activate Pred
    Pred->>Pred: Preprocess (impute, one-hot encode, scale)
    Pred->>Pred: Run model inference (predict & predict_proba)
    Pred->>Pred: Map predicted target using label map & threshold
    Pred-->>UI: Return prediction label & risk probability
    deactivate Pred
    UI->>User: Display Risk State (Positive/Negative) & Probability Score
```

---

## 5. Entity Relationship Diagram of the Proposed System
This diagram shows the logical database schemas, types, and attributes modeling patients, clinical records, symptoms, and generated model predictions.

```mermaid
erDiagram
    PATIENT {
        int patient_id PK
        float age
        string sex
        string residence_area
    }
    CLINICAL_RECORD {
        int record_id PK
        int patient_id FK
        datetime admission_date
        datetime discharge_date
        int length_of_stay
    }
    SYMPTOMS {
        int symptom_id PK
        int record_id FK
        int fever
        int headache
        int abdominal_pain
        int general_body_malaise
        int dizziness
        int vomiting
        int confusion
        int backache
        int chest_pain
        int coughing
        int joint_pain
    }
    PREDICTION {
        int prediction_id PK
        int record_id FK
        int predicted_target
        float probability
        string model_used
        float decision_threshold
        datetime date_created
    }

    PATIENT ||--o{ CLINICAL_RECORD : "undergoes"
    CLINICAL_RECORD ||--|| SYMPTOMS : "presents"
    CLINICAL_RECORD ||--o| PREDICTION : "generates"
```

---

## 6. Conceptual Framework of the Study
This diagram depicts the progression from Demographic and Clinical Symptom Input Variables, through Data Preprocessing, Leakage Prevention, and Class-Imbalance Correction, to Machine Learning Model Training and Comparison, culminating in Malaria Occurrence Prediction, Feature Importance Analysis, and Result Visualization.

```mermaid
flowchart TD
    Input["1. Demographic & Clinical Symptom Input Variables<br>(Age, Sex, Residence, Stay Duration, Fever, Malaise, etc.)"]
    Preproc["2. Data Preprocessing<br>(Normalization, Imputation, One-Hot Encoding, Scaling)"]
    Leakage["3. Leakage Prevention<br>(Train-Test Split, Cross-Validation, Feature Selection)"]
    Imbalance["4. Class-Imbalance Correction<br>(SMOTE, Random Oversampling, Class Weighting)"]
    Training["5. Machine Learning Model Training & Comparison<br>(LR, DT, RF, SVM; Evaluation via F1-Score, Accuracy, ROC-AUC)"]
    Prediction["6. Malaria Occurrence Prediction<br>(Binary Classification: Positive/Negative with Probability)"]
    Feature["7. Feature Importance Analysis<br>(SHAP Values, Permutation Importance, Coefficient Analysis)"]
    Visualization["8. Result Visualization<br>(ROC Curves, Confusion Matrices, Performance Charts, Feature Plots)"]

    Input --> Preproc --> Leakage --> Imbalance --> Training --> Prediction --> Feature --> Visualization
    Training -.-> Feature
    Training -.-> Visualization
    Feature -.-> Visualization
```
