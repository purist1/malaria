"""Feature definitions for the Malaria Occurrence Prediction System.

Updated for the Africa-wide synthetic dataset
(electricsheepafrica/africa-synth-malaria-malaria-dataset-all):

  - Demographic / geographic: age_years, sex, residence (Urban/Rural)
  - Environmental:            season (Rainy/Dry)
  - Preventive:               uses_mosquito_net (bool → 0/1)
  - Clinical lab:             hemoglobin_g_dl (optional, defaults to median)
  - Clinical duration:        fever_days
  - Symptom flags (binary):   has_fever, has_chills, has_headache,
                              has_vomiting, has_diarrhea, has_weakness
  - Target:                   malaria_status → Positive=1, Negative=0

Dropped leakage / post-diagnosis columns:
  patient_id, age_months, age_group,
  parasitemia_level, parasitemia_count, plasmodium_species,
  hemoglobin_g_dl (only if truly unavailable — kept as optional here),
  anemia_status, outcome, malaria_probability_score,
  severe_malaria, cerebral_malaria, respiratory_distress,
  shock, acute_kidney_injury
"""

from __future__ import annotations

# Continuous numeric features (standardised with StandardScaler)
NUMERIC_FEATURES = [
    "age_years",
    "hemoglobin_g_dl",   # optional lab value — defaults to median if missing
    "fever_days",
]

# Binary flags: preventive behaviour + symptom indicators (0/1)
# Included in numeric block so they are scaled alongside continuous features.
SYMPTOM_FEATURES = [
    "uses_mosquito_net",   # 0 = No, 1 = Yes (preventive)
    "has_fever",
    "has_chills",
    "has_headache",
    "has_vomiting",
    "has_diarrhea",
    "has_weakness",
]

# Categorical features (One-Hot Encoded)
CATEGORICAL_FEATURES = [
    "sex",        # Male / Female
    "residence",  # Urban / Rural  (replaces the 5 Kogi-specific LGAs)
    "season",     # Rainy / Dry
]

# Target column name (after normalisation)
TARGET_COLUMN = "malaria_status"

# All numeric columns fed to the scaler (continuous + binary flags)
ALL_NUMERIC_FEATURES = NUMERIC_FEATURES + SYMPTOM_FEATURES

# Combined ordered feature set (order matches preprocessing pipeline output)
ALL_FEATURES = CATEGORICAL_FEATURES + ALL_NUMERIC_FEATURES

# Human-readable labels mapped to target integers
LABEL_MAP = {
    0: "Malaria Negative",
    1: "Malaria Positive",
}

# Raw string target values → integer encoding
TARGET_VALUE_MAP = {
    "positive": 1,
    "1": 1,
    "negative": 0,
    "0": 0,
}

# Columns to drop from the raw HF dataset (leakage / administrative)
HF_DROP_COLUMNS = {
    "patient_id",
    "age_months",
    "age_group",
    "parasitemia_level",
    "parasitemia_count",
    "plasmodium_species",
    "anemia_status",
    "outcome",
    "malaria_probability_score",
    "severe_malaria",
    "cerebral_malaria",
    "respiratory_distress",
    "shock",
    "acute_kidney_injury",
}
