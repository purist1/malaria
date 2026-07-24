"""Streamlit dashboard for the Malaria Occurrence Prediction System."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

# Add project root to sys.path so we can import src
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.malaria_forecast.artifacts import load_artifact
from src.malaria_forecast.config import load_config
from src.malaria_forecast.predict import predict_single_record
from src.malaria_forecast.train import train_all_models

# Configure local logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load config path
CONFIG_PATH = ROOT / "config" / "config.yaml"


def main() -> None:
    try:
        config = load_config(CONFIG_PATH)
    except Exception as exc:
        st.error(f"Failed to load configuration file: {exc}")
        return

    dashboard_cfg = config["dashboard"]
    artifacts_cfg = config["artifacts"]
    data_cfg = config["data"]

    models_dir = ROOT / artifacts_cfg["models_dir"]
    reports_dir = ROOT / artifacts_cfg["reports_dir"]
    figures_dir = ROOT / artifacts_cfg["figures_dir"]

    # Page Configuration with Premium Icons
    st.set_page_config(
        page_title=dashboard_cfg["title"],
        page_icon="🦟",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Injected Premium CSS Custom Styles for Visual Excellence (Aesthetics)
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
        
        /* Font styling override */
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }

        .main-header {
            background: linear-gradient(135deg, #FF4B4B 0%, #8C0000 100%);
            padding: 2.5rem;
            border-radius: 16px;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(140, 0, 0, 0.15);
            text-align: center;
        }
        .main-title {
            font-size: 2.75rem;
            font-weight: 800;
            margin-bottom: 0.5rem;
            letter-spacing: -0.5px;
        }
        .main-subtitle {
            font-size: 1.15rem;
            opacity: 0.9;
            font-weight: 300;
        }
        .card-positive {
            padding: 1.5rem;
            border-radius: 12px;
            background: linear-gradient(135deg, #FFF5F5 0%, #FED7D7 100%);
            border: 2px solid #E53E3E;
            color: #9B2C2C;
            margin-top: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 15px rgba(229, 62, 62, 0.1);
        }
        .card-negative {
            padding: 1.5rem;
            border-radius: 12px;
            background: linear-gradient(135deg, #F0FFF4 0%, #C6F6D5 100%);
            border: 2px solid #38A169;
            color: #22543D;
            margin-top: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 15px rgba(56, 161, 105, 0.1);
        }
        .metric-label-val {
            font-size: 1.85rem;
            font-weight: 800;
        }
        .footer-container {
            margin-top: 3rem;
            padding: 1.5rem;
            background-color: #F7FAFC;
            border-top: 1px solid #E2E8F0;
            border-radius: 8px;
            text-align: center;
            font-size: 0.9rem;
            color: #718096;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # 1. Header Section
    st.markdown(
        f"""
        <div class="main-header">
            <div class="main-title">🦟 {dashboard_cfg["title"]}</div>
            <div class="main-subtitle">An evidence-based clinical decision support tool using machine learning to predict Sub-Saharan Africa patient-level malaria risk.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Load Pipeline Metadata
    metadata_path = models_dir / "metadata.joblib"
    metadata: dict[str, Any] | None = None
    try:
        metadata = load_artifact(metadata_path)
    except FileNotFoundError:
        st.warning("No model metadata found. Please click 'Retrain Models' in the sidebar to train models.")

    # 2. Sidebar - Model & Threshold Controls
    st.sidebar.header("🕹️ Controls & Operations")

    available_models = ["Best Model (Auto-selected)"]
    if metadata is not None:
        model_keys = ["logistic_regression", "decision_tree", "random_forest", "svm"]
        available_models.extend(model_keys)

    selected_model_choice = st.sidebar.selectbox(
        "Inference Model",
        options=available_models,
        help="Select which classifier model to run for single patient risk predictions."
    )

    if selected_model_choice == "Best Model (Auto-selected)":
        selected_model_name = None
    else:
        selected_model_name = selected_model_choice

    # Decision threshold slider
    default_thresh = float(dashboard_cfg.get("default_threshold", 0.5))
    decision_threshold = st.sidebar.slider(
        "Decision Threshold",
        min_value=0.0,
        max_value=1.0,
        value=default_thresh,
        step=0.05,
        help="Decision boundary probability limit. Probability values above this classify the patient as Malaria Positive."
    )

    st.sidebar.divider()
    st.sidebar.subheader("🔄 Retrain Pipeline")
    st.sidebar.caption("Click below to run the preprocessing, training, and metrics generation pipeline end-to-end.")
    
    retrain_clicked = st.sidebar.button("Retrain Models", use_container_width=True)

    if retrain_clicked:
        with st.spinner("Executing model training orchestration pipeline..."):
            try:
                metrics_df, best_name = train_all_models(CONFIG_PATH)
                st.sidebar.success(f"Retraining complete! Best model: {best_name}")
                metadata = load_artifact(metadata_path)
                st.cache_data.clear()
            except Exception as exc:
                st.sidebar.error(f"Retraining failed: {exc}")

    # Tabs definition
    tab1, tab2, tab3 = st.tabs(["🔬 Single Prediction", "📊 Model Comparison", "📋 Dataset Preview"])

    # --- TAB 1: SINGLE PREDICTION ---
    with tab1:
        if metadata is None:
            st.info("Please retrain the models in the sidebar to load the prediction form.")
        else:
            st.subheader("🔬 Single Patient Clinical Risk Form")
            st.caption("Provide patient demographics, environmental context, and clinical symptoms to predict risk.")

            imputations = metadata["imputation_defaults"]

            with st.form("single_prediction_form"):
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Demographic & Environmental Context**")
                    
                    # Sex
                    default_sex = imputations.get("sex", "Male")
                    sex_options = ["Male", "Female"]
                    sex_index = sex_options.index(default_sex) if default_sex in sex_options else 0
                    sex = st.selectbox(
                        "Patient Sex",
                        options=sex_options,
                        index=sex_index,
                        help="Biological sex of the patient."
                    )

                    # Residence
                    default_res = imputations.get("residence", "Rural")
                    res_options = ["Rural", "Urban"]
                    res_index = res_options.index(default_res) if default_res in res_options else 0
                    residence = st.selectbox(
                        "Residence Setting",
                        options=res_options,
                        index=res_index,
                        help="General residential location classification."
                    )

                    # Season
                    default_season = imputations.get("season", "Rainy")
                    season_options = ["Rainy", "Dry"]
                    season_index = season_options.index(default_season) if default_season in season_options else 0
                    season = st.selectbox(
                        "Transmission Season",
                        options=season_options,
                        index=season_index,
                        help="Current regional climate season."
                    )

                    # Age
                    age_years = st.number_input(
                        "Patient Age (Years)",
                        min_value=0.0,
                        max_value=120.0,
                        value=float(imputations.get("age_years", 8.0)),
                        step=1.0,
                        help="Patient's age in years."
                    )

                    # Fever Days
                    fever_days = st.number_input(
                        "Fever Duration (Days)",
                        min_value=0.0,
                        max_value=30.0,
                        value=float(imputations.get("fever_days", 1.0)),
                        step=1.0,
                        help="Number of consecutive days patient experienced fever (0 if no fever)."
                    )

                    # Mosquito net usage
                    default_net = int(imputations.get("uses_mosquito_net", 0))
                    uses_net_str = st.selectbox(
                        "Uses Insecticide-Treated Mosquito Net",
                        options=["No", "Yes"],
                        index=default_net,
                        help="Whether the patient regularly sleeps under a mosquito net."
                    )
                    uses_mosquito_net = 1 if uses_net_str == "Yes" else 0

                    st.markdown("**Optional Laboratory Parameters**")
                    has_hb = st.checkbox("Provide Hemoglobin (Hb) Test Result", value=True)
                    if has_hb:
                        hb_val = st.number_input(
                            "Hemoglobin Level (g/dL)",
                            min_value=2.0,
                            max_value=22.0,
                            value=float(imputations.get("hemoglobin_g_dl", 11.4)),
                            step=0.1,
                            help="Measured blood hemoglobin level in g/dL."
                        )
                        hemoglobin_g_dl = hb_val
                    else:
                        hemoglobin_g_dl = float(imputations.get("hemoglobin_g_dl", 11.4))

                with col2:
                    st.markdown("**Clinical Symptoms**")
                    symptoms = [
                        ("has_fever", "Fever"),
                        ("has_chills", "Chills / Rigors"),
                        ("has_headache", "Headache"),
                        ("has_vomiting", "Vomiting / Nausea"),
                        ("has_diarrhea", "Diarrhea"),
                        ("has_weakness", "General Body Weakness / Fatigue"),
                    ]
                    
                    symptom_inputs = {}
                    for key, label in symptoms:
                        default_val = int(imputations.get(key, 0))
                        ans = st.selectbox(
                            label,
                            options=["No", "Yes"],
                            index=default_val,
                            help=f"Does the patient present with {label.lower()}?"
                        )
                        symptom_inputs[key] = 1 if ans == "Yes" else 0

                submit_prediction = st.form_submit_button("Predict Malaria Risk", use_container_width=True)

            if submit_prediction:
                record = {
                    "sex": sex,
                    "residence": residence,
                    "season": season,
                    "age_years": age_years,
                    "fever_days": fever_days,
                    "uses_mosquito_net": uses_mosquito_net,
                    "hemoglobin_g_dl": hemoglobin_g_dl,
                    **symptom_inputs
                }

                try:
                    res = predict_single_record(
                        record=record,
                        model_dir=models_dir,
                        model_name=selected_model_name
                    )

                    prob = float(res["probability"])
                    
                    is_positive = prob >= decision_threshold
                    res_label = "Malaria Positive" if is_positive else "Malaria Negative"
                    card_class = "card-positive" if is_positive else "card-negative"

                    if prob == 0.0:
                        prob_str = "0.00%"
                    elif prob < 0.0001:
                        prob_str = "< 0.01%"
                    else:
                        prob_str = f"{prob:.2%}"

                    st.markdown(
                        f"""
                        <div class="{card_class}">
                            <h3>Risk State: {res_label}</h3>
                            <p>Model Classifier Used: <b>{res["model_used"]}</b></p>
                            <p class="metric-label-val">Predicted Risk Probability: {prob_str}</p>
                            <p>Decision Threshold Configured: {decision_threshold:.0%}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.progress(prob)

                except Exception as exc:
                    st.error(f"Error running inference prediction: {exc}")

    # --- TAB 2: MODEL COMPARISON ---
    with tab2:
        st.subheader("📊 Classifier Performance & Metric Comparison")
        st.caption("A summary comparing the performance metrics of the four trained models.")

        comp_csv_path = reports_dir / "model_comparison.csv"
        if not comp_csv_path.exists():
            st.info("Comparison metrics are unavailable. Please retrain the models in the sidebar first.")
        else:
            metrics_df = pd.read_csv(comp_csv_path)

            col_met_1, col_met_2 = st.columns([1, 1])

            with col_met_1:
                st.markdown("#### Performance Metric Scores")
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)

            with col_met_2:
                best_model_row = metrics_df.iloc[0]
                st.markdown(
                    f"""
                    <div style="background-color: #F7FAFC; padding: 1.5rem; border-radius: 12px; border: 1px solid #E2E8F0;">
                        <h4>🌟 Pipeline Selected Best Model</h4>
                        <p style="font-size: 1.5rem; font-weight: 800; color: #2B6CB0; margin-bottom:0.25rem;">{best_model_row['model']}</p>
                        <p style="margin: 0;">F1-Score: <b>{best_model_row['f1_score']:.4f}</b></p>
                        <p style="margin: 0;">ROC-AUC: <b>{best_model_row['roc_auc']:.4f}</b></p>
                        <p style="margin: 0;">Accuracy: <b>{best_model_row['accuracy']:.4f}</b></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.divider()

            col_fig_1, col_fig_2 = st.columns(2)
            with col_fig_1:
                roc_img = figures_dir / "roc_comparison.png"
                if roc_img.exists():
                    st.image(str(roc_img), caption="Combined ROC Curves Comparison", use_column_width=True)

            with col_fig_2:
                bar_img = figures_dir / "metric_comparison_bar.png"
                if bar_img.exists():
                    st.image(str(bar_img), caption="Model Metrics Summary Bar Chart", use_column_width=True)

            st.divider()
            st.markdown("#### Model Confusion Matrices")
            
            grid_col1, grid_col2 = st.columns(2)
            model_keys = ["logistic_regression", "decision_tree", "random_forest", "svm"]
            readable_model_names = {
                "logistic_regression": "Logistic Regression",
                "decision_tree": "Decision Tree",
                "random_forest": "Random Forest",
                "svm": "SVM"
            }

            for idx, model_key in enumerate(model_keys):
                cm_img = figures_dir / f"confusion_matrix_{model_key}.png"
                if cm_img.exists():
                    col_to_use = grid_col1 if idx % 2 == 0 else grid_col2
                    with col_to_use:
                        st.image(
                            str(cm_img),
                            caption=f"Confusion Matrix - {readable_model_names[model_key]}",
                            use_column_width=True
                        )

    # --- TAB 3: DATASET PREVIEW ---
    with tab3:
        st.subheader("📋 Dataset Preview & Quality Statistics")
        processed_data_path = ROOT / data_cfg["processed_path"]
        dataset_path = processed_data_path if processed_data_path.exists() else ROOT / data_cfg["raw_path"]

        if not dataset_path.exists():
            st.info("Dataset file is missing.")
        else:
            data_df = pd.read_csv(dataset_path)

            st.markdown(f"Currently displaying: `{dataset_path.name}` (shape: {data_df.shape[0]} rows, {data_df.shape[1]} columns)")
            st.dataframe(data_df.head(100), use_container_width=True)

            st.divider()

            col_stats1, col_stats2 = st.columns([2, 1])

            with col_stats1:
                st.markdown("#### Feature Descriptive Statistics")
                st.dataframe(data_df.describe().transpose(), use_container_width=True)

            with col_stats2:
                st.markdown("#### Class Balance Distribution")
                target_col = "malaria_status" if "malaria_status" in data_df.columns else ("target" if "target" in data_df.columns else "Result")
                
                if target_col in data_df.columns:
                    counts = data_df[target_col].value_counts()
                    pcts = data_df[target_col].value_counts(normalize=True)
                    
                    dist_df = pd.DataFrame({
                        "Count": counts,
                        "Percentage": pcts.map(lambda p: f"{p:.2%}")
                    })
                    st.dataframe(dist_df, use_container_width=True)
                else:
                    st.caption(f"Target column '{target_col}' not found in dataset columns.")

    # 4. Footer Section
    st.markdown(
        """
        <div class="footer-container">
            ⚠️ <b>Clinical Support Disclaimer:</b> This system is designed solely as a clinical decision-support tool. 
            All predictions must be validated by trained medical specialists and laboratory diagnostic procedures 
            (e.g., blood smear micro-examination, rapid diagnostic tests).
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()