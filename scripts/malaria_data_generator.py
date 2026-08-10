#!/usr/bin/env python3
"""Generate realistic synthetic malaria dataset with configurable noise.

This script creates a synthetic malaria dataset that avoids deterministic patterns
by adding controlled noise to feature-target relationships, making it more suitable
for realistic model evaluation.

Usage:
    python scripts/malaria_data_generator.py -n 10000 -p 0.45 -s 42 --noise 0.15
"""

import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def generate_malaria_dataset(
    n_samples: int = 10000,
    positive_prob: float = 0.45,
    random_seed: int = 42,
    noise_level: float = 0.15
) -> pd.DataFrame:
    """Generate synthetic malaria dataset with realistic noise.
    
    Args:
        n_samples: Total number of samples to generate
        positive_prob: Probability of malaria positive class
        random_seed: Random seed for reproducibility
        noise_level: Amount of noise to add (0.0 = deterministic, 1.0 = random)
    
    Returns:
        DataFrame with synthetic malaria data
    """
    np.random.seed(random_seed)
    logger.info(f"Generating {n_samples} samples with noise level {noise_level}")
    
    # Generate demographic features
    age_years = np.random.gamma(shape=2, scale=15, size=n_samples).clip(0, 80)
    sex = np.random.choice(['Male', 'Female'], size=n_samples, p=[0.52, 0.48])
    residence = np.random.choice(['Urban', 'Rural'], size=n_samples, p=[0.4, 0.6])
    season = np.random.choice(['Rainy', 'Dry'], size=n_samples, p=[0.6, 0.4])
    
    # Generate preventive behavior
    uses_mosquito_net = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])
    
    # Generate target variable first
    malaria_status = np.random.choice([0, 1], size=n_samples, p=[1-positive_prob, positive_prob])
    
    # Generate clinical features with realistic relationships + noise
    # Base probability of symptoms given malaria status
    base_fever_prob = np.where(malaria_status == 1, 0.95, 0.3)
    base_chills_prob = np.where(malaria_status == 1, 0.85, 0.25)
    base_headache_prob = np.where(malaria_status == 1, 0.75, 0.4)
    base_vomiting_prob = np.where(malaria_status == 1, 0.45, 0.15)
    base_diarrhea_prob = np.where(malaria_status == 1, 0.35, 0.2)
    base_weakness_prob = np.where(malaria_status == 1, 0.80, 0.35)
    
    # Add noise to symptom probabilities
    noise_factor = np.random.uniform(1-noise_level, 1+noise_level, size=n_samples)
    
    has_fever = (np.random.random(n_samples) < base_fever_prob * noise_factor).astype(int)
    has_chills = (np.random.random(n_samples) < base_chills_prob * noise_factor).astype(int)
    has_headache = (np.random.random(n_samples) < base_headache_prob * noise_factor).astype(int)
    has_vomiting = (np.random.random(n_samples) < base_vomiting_prob * noise_factor).astype(int)
    has_diarrhea = (np.random.random(n_samples) < base_diarrhea_prob * noise_factor).astype(int)
    has_weakness = (np.random.random(n_samples) < base_weakness_prob * noise_factor).astype(int)
    
    # Generate fever days with noise (no longer perfectly deterministic)
    # Base: positive cases have longer fever duration
    base_fever_days = np.where(
        malaria_status == 1,
        np.random.gamma(shape=2, scale=2, size=n_samples),  # Mean ~4 days for positive
        np.random.gamma(shape=1, scale=0.5, size=n_samples)   # Mean ~0.5 days for negative
    )
    
    # Add noise to fever days
    fever_noise = np.random.normal(0, noise_level * 2, size=n_samples)
    fever_days = np.clip(base_fever_days + fever_noise, 0, 14).round().astype(int)
    
    # Generate hemoglobin with noise
    base_hemoglobin = np.where(
        malaria_status == 1,
        np.random.normal(10.5, 1.5, size=n_samples),  # Lower for malaria
        np.random.normal(12.5, 1.2, size=n_samples)   # Normal range
    )
    hemoglobin_noise = np.random.normal(0, noise_level, size=n_samples)
    hemoglobin_g_dl = np.clip(base_hemoglobin + hemoglobin_noise, 5, 18)
    
    # Introduce some missing values (realistic)
    missing_mask = np.random.random(n_samples) < 0.1
    hemoglobin_g_dl[missing_mask] = np.nan
    
    # Create DataFrame
    df = pd.DataFrame({
        'patient_id': range(n_samples),
        'age_years': age_years,
        'age_months': (age_years * 12).astype(int),
        'age_group': pd.cut(age_years, bins=[0, 5, 15, 50, 100], labels=['0-5', '6-15', '16-50', '50+']),
        'sex': sex,
        'residence': residence,
        'season': season,
        'uses_mosquito_net': uses_mosquito_net,
        'malaria_status': np.where(malaria_status == 1, 'Positive', 'Negative'),
        'parasitemia_level': np.where(malaria_status == 1, np.random.randint(1, 5, size=n_samples), 0),
        'parasitemia_count': np.where(malaria_status == 1, np.random.randint(100, 50000, size=n_samples), 0),
        'plasmodium_species': np.where(malaria_status == 1, np.random.choice(['falciparum', 'vivax', 'ovale', 'malariae'], size=n_samples), 'None'),
        'hemoglobin_g_dl': hemoglobin_g_dl,
        'anemia_status': np.where(hemoglobin_g_dl < 11, 'Yes', 'No'),
        'fever_days': fever_days,
        'has_fever': has_fever,
        'has_chills': has_chills,
        'has_headache': has_headache,
        'has_vomiting': has_vomiting,
        'has_diarrhea': has_diarrhea,
        'has_weakness': has_weakness,
        'severe_malaria': np.where((malaria_status == 1) & (np.random.random(n_samples) < 0.15), 'Yes', 'No'),
        'cerebral_malaria': np.where((malaria_status == 1) & (np.random.random(n_samples) < 0.05), 'Yes', 'No'),
        'respiratory_distress': np.where((malaria_status == 1) & (np.random.random(n_samples) < 0.08), 'Yes', 'No'),
        'shock': np.where((malaria_status == 1) & (np.random.random(n_samples) < 0.03), 'Yes', 'No'),
        'acute_kidney_injury': np.where((malaria_status == 1) & (np.random.random(n_samples) < 0.04), 'Yes', 'No'),
        'outcome': np.where(malaria_status == 1, np.random.choice(['Recovered', 'Died'], size=n_samples, p=[0.92, 0.08]), 'Recovered'),
        'malaria_probability_score': np.where(malaria_status == 1, np.random.uniform(0.6, 0.99, size=n_samples), np.random.uniform(0.01, 0.4, size=n_samples))
    })
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic malaria dataset with configurable noise")
    parser.add_argument('-n', '--n-samples', type=int, default=10000, help='Number of samples to generate')
    parser.add_argument('-p', '--positive-prob', type=float, default=0.45, help='Probability of positive class')
    parser.add_argument('-s', '--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--noise', type=float, default=0.15, help='Noise level (0.0 = deterministic, 1.0 = random)')
    parser.add_argument('-o', '--output', type=str, default='dataset/africa_malaria_hf/train.csv', help='Output file path')
    
    args = parser.parse_args()
    
    # Generate dataset
    df = generate_malaria_dataset(
        n_samples=args.n_samples,
        positive_prob=args.positive_prob,
        random_seed=args.seed,
        noise_level=args.noise
    )
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save dataset
    df.to_csv(output_path, index=False)
    logger.info(f"Dataset saved to {output_path}")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Target distribution:\n{df['malaria_status'].value_counts()}")
    
    # Check for deterministic patterns
    neg_fever = df[df['malaria_status'] == 'Negative']['fever_days']
    pos_fever = df[df['malaria_status'] == 'Positive']['fever_days']
    logger.info(f"Negative fever_days - min: {neg_fever.min()}, max: {neg_fever.max()}, mean: {neg_fever.mean():.2f}")
    logger.info(f"Positive fever_days - min: {pos_fever.min()}, max: {pos_fever.max()}, mean: {pos_fever.mean():.2f}")
    
    # Calculate correlation
    fever_corr = df['fever_days'].corr(df['malaria_status'].map({'Positive': 1, 'Negative': 0}))
    logger.info(f"fever_days correlation with target: {fever_corr:.4f}")
    
    if neg_fever.min() == 0 and neg_fever.max() == 0:
        logger.warning("⚠️  Deterministic pattern detected: All negative cases have fever_days = 0")
    else:
        logger.info("✅ Noise successfully added: No deterministic patterns detected")


if __name__ == "__main__":
    main()
