"""Feature definitions for the Malaria Occurrence Prediction System.

Updated for Malaria_Dataset.csv which contains:
  - Demographic: Age, Sex, Residence_Area
  - Clinical symptoms (binary flags): Fever, Headache, Abdominal_Pain,
    General_Body_Malaise, Dizziness, Vomiting, Confusion, Backache,
    Chest_Pain, Coughing, Joint_Pain
  - Engineered: length_of_stay (days, derived from DOA and Discharge_Date)
  - Target: Target (0 = no severe malaria, 1 = severe/confirmed malaria)

Dropped columns (no predictive value / data leakage):
  IP_Number, DOA, Discharge_Date, Primary_Code, Diagnosis_Type, Risk_Score
"""

from __future__ import annotations

# Numeric features: continuous/count columns to be scaled
NUMERIC_FEATURES = [
    "age",
    "length_of_stay",  # engineered: Discharge_Date - DOA in days
]

# Binary symptom flags: already 0/1, included as numeric for scaling
SYMPTOM_FEATURES = [
    "fever",
    "headache",
    "abdominal_pain",
    "general_body_malaise",
    "dizziness",
    "vomiting",
    "confusion",
    "backache",
    "chest_pain",
    "coughing",
    "joint_pain",
]

# Categorical demographic/geographic features (will be one-hot encoded)
CATEGORICAL_FEATURES = [
    "sex",
    "residence_area",
]

# Target column name (after normalization)
TARGET_COLUMN = "target"

# All numeric columns fed to the scaler (continuous + binary symptoms)
ALL_NUMERIC_FEATURES = NUMERIC_FEATURES + SYMPTOM_FEATURES

# Combined feature set (order matches preprocessing pipeline output)
ALL_FEATURES = CATEGORICAL_FEATURES + ALL_NUMERIC_FEATURES

# Human-readable labels mapped to target integers
LABEL_MAP = {
    0: "Malaria Negative / Mild",
    1: "Malaria Positive / Severe",
}

# Raw target values accepted by the loader
TARGET_VALUE_MAP = {
    "1": 1,
    "0": 0,
}
