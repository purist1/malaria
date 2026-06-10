"""Streamlit dashboard for malaria occurrence prediction."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.predict import load_artifacts, predict_dataframe
from src.train import train_all_models

CONFIG_PATH = ROOT / "config" / "config.yaml"


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open() as file:
        return yaml.safe_load(file)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


@st.cache_data(ttl=120)
def load_model_comparison(comparison_path: Path) -> pd.DataFrame | None:
    if not comparison_path.exists():
        return None
    return pd.read_csv(comparison_path)


@st.cache_data(ttl=120)
def load_explainability_report(importance_path: Path) -> pd.DataFrame | None:
    if not importance_path.exists():
        return None
    return pd.read_csv(importance_path)


@st.cache_data(ttl=300)
def load_dataset_preview(dataset_path: Path, rows: int = 30) -> pd.DataFrame:
    return pd.read_csv(dataset_path).head(rows)


@st.cache_data(ttl=120)
def load_latest_experiment(index_path: Path) -> dict[str, Any] | None:
    if not index_path.exists():
        return None
    lines = [line.strip() for line in index_path.read_text().splitlines() if line.strip()]
    if not lines:
        return None
    return json.loads(lines[-1])


def _render_prediction_form(defaults: dict[str, Any]) -> pd.DataFrame | None:
    st.subheader("🔬 Single-Record Prediction")
    st.caption("Provide patient hematology and demographic features to estimate malaria occurrence risk.")

    with st.form("single_prediction_form"):
        col_a, col_b = st.columns(2)

        with col_a:
            default_sex = str(defaults.get("sex", "Male"))
            sex_options = ["Male", "Female"]
            selected_idx = 0 if default_sex.lower().startswith("m") else 1
            sex = st.selectbox(
                "Sex", 
                options=sex_options, 
                index=selected_idx,
                help="Patient's biological sex."
            )
            age = st.number_input(
                "Age (Years)", 
                min_value=0.0, 
                max_value=120.0, 
                value=float(defaults.get("age", 30.0)),
                help="Patient's age in years. Range: 0 to 120."
            )
            hemoglobin_hb_pct = st.number_input(
                "Hemoglobin (Hb%) (g/dL)",
                min_value=0.0,
                max_value=30.0,
                value=float(defaults.get("hemoglobin_hb_pct", 14.0)),
                help="Oxygen-carrying capacity of red blood cells. Typical normal range: 12.0 - 17.5 g/dL. (Allowed bounds: 2.0 - 24.0 g/dL)"
            )
            total_wbc_count_cumm = st.number_input(
                "Total White Blood Cell (WBC) Count (cells/µL)",
                min_value=0.0,
                value=float(defaults.get("total_wbc_count_cumm", 6000.0)),
                step=100.0,
                help="Total volume of leukocyte immune cells. Typical normal range: 4,000 - 11,000 cells/µL. (Allowed bounds: 500 - 120,000 cells/µL)"
            )
            neutrophils = st.number_input(
                "Neutrophils (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(defaults.get("neutrophils", 45.0)),
                help="Percentage of neutrophils (bacterial defense cells) among total WBCs. Typical normal range: 40% - 70%. (Allowed bounds: 0% - 100%)"
            )
            lymphocytes = st.number_input(
                "Lymphocytes (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(defaults.get("lymphocytes", 40.0)),
                help="Percentage of lymphocytes (viral and specific defense cells) among total WBCs. Typical normal range: 20% - 40%. (Allowed bounds: 0% - 100%)"
            )

        with col_b:
            total_cir_eosinophils = st.number_input(
                "Total Circulating Eosinophils (cells/µL)",
                min_value=0.0,
                value=float(defaults.get("total_cir_eosinophils", 280.0)),
                step=5.0,
                help="Immune cells activated during parasitic infections and allergic responses. Typical normal range: 50 - 500 cells/µL. (Allowed bounds: 0 - 5,000 cells/µL)"
            )
            htc_pcv_pct = st.number_input(
                "Hematocrit / Packed Cell Volume (HCT/PCV %)",
                min_value=0.0,
                max_value=100.0,
                value=float(defaults.get("htc_pcv_pct", 46.0)),
                help="Volume percentage of red blood cells in whole blood. Typical normal range: 36% - 50%. (Allowed bounds: 10% - 70%)"
            )
            mch_pg = st.number_input(
                "Mean Corpuscular Hemoglobin (MCH) (pg)",
                min_value=0.0,
                value=float(defaults.get("mch_pg", 30.0)),
                help="Average mass of hemoglobin per individual red blood cell. Typical normal range: 27 - 33 pg. (Allowed bounds: 10 - 45 pg)"
            )
            mchc_g_dl = st.number_input(
                "Mean Corpuscular Hemoglobin Concentration (MCHC) (g/dL)",
                min_value=0.0,
                value=float(defaults.get("mchc_g_dl", 31.0)),
                help="Average concentration of hemoglobin in a given volume of packed red blood cells. Typical normal range: 32 - 36 g/dL. (Allowed bounds: 20 - 45 g/dL)"
            )
            rdw_cv_pct = st.number_input(
                "Red Cell Distribution Width (RDW-CV %)",
                min_value=0.0,
                value=float(defaults.get("rdw_cv_pct", 15.0)),
                help="Indicates standard deviation of red blood cell sizes. High values reflect high size variation. Typical normal range: 11.5% - 14.5%. (Allowed bounds: 5% - 35%)"
            )
            platelet_count = st.number_input(
                "Platelet Count (cells/µL)",
                min_value=0.0,
                value=float(defaults.get("platelet_count", 140000.0)),
                step=1000.0,
                help="Volume of platelets responsible for clotting and coagulation. Typical normal range: 150,000 - 450,000 cells/µL. (Allowed bounds: 5,000 - 1,500,000 cells/µL)"
            )

        submitted = st.form_submit_button("Submit Patient Record for Risk Prediction", use_container_width=True)

    if not submitted:
        return None

    return pd.DataFrame(
        [
            {
                "sex": sex,
                "age": age,
                "hemoglobin_hb_pct": hemoglobin_hb_pct,
                "total_wbc_count_cumm": total_wbc_count_cumm,
                "neutrophils": neutrophils,
                "lymphocytes": lymphocytes,
                "total_cir_eosinophils": total_cir_eosinophils,
                "htc_pcv_pct": htc_pcv_pct,
                "mch_pg": mch_pg,
                "mchc_g_dl": mchc_g_dl,
                "rdw_cv_pct": rdw_cv_pct,
                "platelet_count": platelet_count,
            }
        ]
    )


def _render_prediction_output(result: pd.Series, threshold: float) -> None:
    probability = float(result.get("probability", 0.0))
    predicted_positive = int(result["prediction"]) == 1
    risk_state = "High Risk" if probability >= threshold else "Lower Risk"

    if predicted_positive:
        st.markdown(
            f"""
            <div style="padding: 1.2rem; border-radius: 8px; background-color: #fff5f5; border: 1px solid #feb2b2; margin-bottom: 1.5rem;">
                <span style="font-size: 1.25rem; font-weight: 700; color: #c53030;">⚠️ Prediction: {result['label']}</span>
                <p style="margin: 0.5rem 0 0 0; color: #742a2a; font-size: 0.95rem;">
                    Patient parameters cross the defined risk threshold. Urgent clinical follow-up is recommended.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div style="padding: 1.2rem; border-radius: 8px; background-color: #f0fff4; border: 1px solid #9ae6b4; margin-bottom: 1.5rem;">
                <span style="font-size: 1.25rem; font-weight: 700; color: #22543d;">✅ Prediction: {result['label']}</span>
                <p style="margin: 0.5rem 0 0 0; color: #22543d; font-size: 0.95rem;">
                    Patient parameters indicate low risk of malaria occurrence based on the clinical profile.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    metric_col_a, metric_col_b, metric_col_c = st.columns(3)
    metric_col_a.metric("Predicted probability", f"{probability:.2%}")
    metric_col_b.metric("Configured threshold", f"{threshold:.0%}")
    metric_col_c.metric("Risk state", risk_state)

    st.progress(min(max(probability, 0.0), 1.0))
    st.caption("Decision support only — clinical diagnosis must be made by qualified professionals.")


def _render_model_summary_cards(comparison_df: pd.DataFrame) -> None:
    best = comparison_df.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best model", str(best["model"]))
    c2.metric("Accuracy", f"{float(best['accuracy']):.3f}")
    c3.metric("F1-score", f"{float(best['f1_score']):.3f}")
    roc_value = best.get("roc_auc")
    c4.metric("ROC-AUC", "N/A" if pd.isna(roc_value) else f"{float(roc_value):.3f}")


def _render_homepage(comparison_df: pd.DataFrame | None) -> None:
    st.markdown(
        """
        <div class="hero-container">
            <div class="hero-title">🦟 Malaria Occurrence Prediction System</div>
            <div class="hero-subtitle">An AI-powered clinical decision-support tool leveraging machine learning to predict malaria risk based on patient hematology profiles.</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("Welcome to the Clinical Predictor")
        st.write(
            """
            This system is designed to analyze complete blood count (CBC) parameters and patient demographics 
            to assist medical professionals in identifying patients at high risk of malaria. By comparing 
            four standard machine learning models—**Logistic Regression**, **Decision Tree**, **Random Forest**, 
            and **SVM**—the tool selects the top-performing algorithm to perform fast, evidence-based risk assessment.
            """
        )
        
        st.markdown("### Quick Navigation")
        c1, c2 = st.columns(2)
        with c1:
            st.info("**Run Predictions**\n\nSubmit individual patient CBC records or process high-throughput clinical records in batches via CSV upload.")
            if st.button("🔬 Start Patient Predictor", use_container_width=True):
                st.session_state.page = "🔬 Run Predictor"
                st.rerun()
        with c2:
            st.success("**How It Works**\n\nLearn about the medical science, feature sets, preprocessing pipelines, and model architectures behind this project.")
            if st.button("📖 Explore Methodology", use_container_width=True):
                st.session_state.page = "📖 How It Works"
                st.rerun()
                
    with col_right:
        st.subheader("Model Performance")
        if comparison_df is not None:
            best_model = comparison_df.iloc[0]
            
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Active Model</div>
                    <div class="metric-value" style="font-size: 1.4rem; color: #3182ce; font-weight: bold;">{best_model['model']}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Accuracy</div>
                    <div class="metric-value">{float(best_model['accuracy']):.1%}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">F1-Score</div>
                    <div class="metric-value">{float(best_model['f1_score']):.1%}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.warning("No model trained yet. Go to Database & Ops page to retrain models.")
            if st.button("⚙️ Go to Operations", use_container_width=True, key="go_to_ops_home"):
                st.session_state.page = "⚙️ Database & Ops"
                st.rerun()
                
    st.divider()
    st.markdown(
        """
        > ⚠️ **Clinical Disclaimer & Ethical considerations:**
        > This tool is developed for medical research and decision support purposes only. It is **not a replacement** 
        > for laboratory diagnosis (e.g., blood smears, rapid diagnostic tests (RDTs)) or professional medical consults. 
        > All predictions must be validated by trained clinical personnel. Patient privacy must be handled according to 
        > local health data regulations.
        """
    )


def _render_how_it_works() -> None:
    st.header("📖 System Methodology & How It Works")
    st.caption("A scientific breakdown of the data processing, feature engineering, and model training workflow.")
    
    st.markdown("### End-to-End Prediction Pipeline")
    
    st.markdown(
        """
        <div class="step-container">
            <div class="step-number">Step 1</div>
            <div class="step-title">Clinical & Demographic Data Inputs</div>
            <p>The system consumes standard patient demographics (Age, Sex) and CBC (Complete Blood Count) features. 
            These clinical variables capture physiological responses to malaria infection, such as anemia (indicated by low Hemoglobin and Hematocrit) 
            and thrombocytopenia (indicated by low Platelet count).</p>
        </div>
        <div class="step-container">
            <div class="step-number">Step 2</div>
            <div class="step-title">Validation & Preprocessing</div>
            <p>Input data is run through strict medical-range validation checks to flag anomalous values. Missing numeric features are filled 
            using the dataset median, categorical features using the mode, and duplicates are dropped. Numerical columns are standardized using 
            a StandardScaler, and categorical inputs are encoded using One-Hot encoding.</p>
        </div>
        <div class="step-container">
            <div class="step-number">Step 3</div>
            <div class="step-title">Multi-Model Training & Cross-Validation</div>
            <p>Four distinct algorithms are trained and compared under stratified configurations:
            <ul>
                <li><b>Logistic Regression</b>: Provides a baseline linear decision boundary.</li>
                <li><b>Decision Tree</b>: Provides interpretable rule-based partitions.</li>
                <li><b>Random Forest</b>: Reduces variance via an ensemble of decision trees.</li>
                <li><b>SVM</b>: Projects features into high-dimensional space for non-linear boundaries.</li>
            </ul>
            Models are evaluated based on <b>F1-Score</b>, balancing Precision (minimizing false alarms) and Recall (minimizing missed cases).</p>
        </div>
        <div class="step-container">
            <div class="step-number">Step 4</div>
            <div class="step-title">Inference & Probability Calibration</div>
            <p>During inference, the top-performing model is loaded. For the SVM, probabilities are calibrated using Platt scaling 
            (sigmoid method) to output reliable risk probabilities rather than binary outputs. Predictions exceeding the configured threshold 
            are flagged as High Risk / Malaria Positive.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.subheader("Key Clinical Indicators")
    st.markdown(
        """
        Studies show that several Complete Blood Count (CBC) parameters are highly correlated with malaria infections:
        - **Platelets:** Thrombocytopenia (low platelets) is one of the most common hematological changes in malaria.
        - **Hemoglobin & Hematocrit (PCV):** Malaria parasites invade and destroy red blood cells, leading to hemolytic anemia, which lowers hemoglobin and hematocrit percentage.
        - **Lymphocytes & Neutrophils:** Parasite load shifts white blood cell ratios, resulting in changes in neutrophil/lymphocyte counts as part of the immune response.
        - **RDW-CV:** Variation in red cell width increases as the bone marrow releases immature red cells to replace those destroyed by the parasite.
        """
    )
    
    st.subheader("Academic References")
    st.markdown(
        """
        - WHO (2025). *World Malaria Report 2024*. World Health Organization.
        - Nkiruka et al. (2020). Prediction of malaria incidence using climate variables. *Informatics in Medicine Unlocked*.
        - Ahmed et al. (2021). Random Forest and Logistic Regression for malaria prediction. *Journal of Healthcare Informatics*.
        - Akinyemi et al. (2021). Malaria prediction using Random Forest Algorithm. *Applied AI in Medicine*.
        """
    )


def main() -> None:
    config = load_config()
    dashboard_cfg = config.get("dashboard", {})
    artifacts_cfg = config.get("artifacts", {})
    data_cfg = config.get("data", {})
    validation_cfg = config.get("validation", {})

    model_dir = resolve_path(artifacts_cfg.get("model_dir", "models"))
    reports_dir = resolve_path(artifacts_cfg.get("reports_dir", "reports"))
    comparison_path = reports_dir / "model_comparison.csv"
    explainability_path = reports_dir / "explainability" / "permutation_importance.csv"
    experiments_index = reports_dir / "experiments" / "index.jsonl"
    dataset_path = resolve_path(data_cfg.get("dataset_path", "dataset/Malaria Diseases dataset - .csv"))

    st.set_page_config(
        page_title=dashboard_cfg.get("title", "Malaria Occurrence Prediction System"),
        page_icon="🦟",
        layout="wide",
    )

    # Injected Premium CSS Custom Styles
    st.markdown(
        """
        <style>
        .hero-container {
            background: linear-gradient(135deg, #1A365D 0%, #2A4365 100%);
            padding: 2.5rem;
            border-radius: 12px;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .hero-title {
            font-size: 2.25rem;
            font-weight: 800;
            margin-bottom: 0.75rem;
        }
        .hero-subtitle {
            font-size: 1.1rem;
            opacity: 0.9;
            line-height: 1.5;
        }
        .metric-card {
            background: #F7FAFC;
            border: 1px solid #E2E8F0;
            border-radius: 8px;
            padding: 1.25rem;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        }
        .metric-title {
            font-size: 0.8rem;
            font-weight: bold;
            color: #718096;
            text-transform: uppercase;
            margin-bottom: 0.25rem;
        }
        .metric-value {
            font-size: 1.75rem;
            font-weight: 800;
            color: #2D3748;
        }
        .step-container {
            border-left: 4px solid #3182CE;
            padding-left: 1.25rem;
            margin-bottom: 1.5rem;
        }
        .step-number {
            font-size: 0.9rem;
            font-weight: 800;
            color: #3182CE;
            text-transform: uppercase;
        }
        .step-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: #2D3748;
            margin: 0.25rem 0 0.5rem 0;
        }
        @media (prefers-color-scheme: dark) {
            .metric-card {
                background: #1A202C;
                border-color: #2D3748;
            }
            .metric-value {
                color: #EDF2F7;
            }
            .step-title {
                color: #EDF2F7;
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Set up navigation state
    if "page" not in st.session_state:
        st.session_state.page = "🏠 Home"

    # Sidebar Navigation and Control Panel
    st.sidebar.header("Navigation")
    sidebar_page = st.sidebar.radio(
        "Select Page",
        options=["🏠 Home", "📖 How It Works", "🔬 Run Predictor", "📊 Model Insights", "⚙️ Database & Ops"],
        index=["🏠 Home", "📖 How It Works", "🔬 Run Predictor", "📊 Model Insights", "⚙️ Database & Ops"].index(st.session_state.page),
        label_visibility="collapsed"
    )
    st.session_state.page = sidebar_page

    st.sidebar.divider()
    
    default_threshold = float(dashboard_cfg.get("positive_threshold", 0.5))
    st.sidebar.header("Operational Controls")
    threshold = st.sidebar.slider(
        "Decision Threshold",
        min_value=0.1,
        max_value=0.9,
        value=default_threshold,
        step=0.05,
        help="Decision probability limit. Probabilities exceeding this threshold classify as Malaria Positive."
    )
    train_now = st.sidebar.button("Train / Retrain Models", use_container_width=True)

    metadata: dict[str, Any] | None = None
    available_models: list[str] = []
    try:
        _, metadata = load_artifacts(model_dir=model_dir)
        available_models = sorted((metadata.get("model_files") or {}).keys())
    except FileNotFoundError:
        metadata = None

    model_choice = st.sidebar.selectbox(
        "Inference Model",
        options=["Best Model (Auto)"] + available_models,
    )
    selected_model_name = None if model_choice == "Best Model (Auto)" else model_choice

    if train_now:
        with st.spinner("Training models and generating artifacts..."):
            results_df, metadata = train_all_models(config_path=CONFIG_PATH)
        st.cache_data.clear()
        st.sidebar.success(f"Training completed. Best model: {metadata['best_model_name']}")
        st.sidebar.dataframe(results_df, hide_index=True, use_container_width=True)
        available_models = sorted((metadata.get("model_files") or {}).keys())
        model_choice = st.sidebar.selectbox(
            "Inference Model (Post-Train)",
            options=["Best Model (Auto)"] + available_models,
            key="post_train_model_choice",
        )
        selected_model_name = None if model_choice == "Best Model (Auto)" else model_choice

    comparison_df = load_model_comparison(comparison_path)

    # PAGE ROUTING
    if st.session_state.page == "🏠 Home":
        _render_homepage(comparison_df)

    elif st.session_state.page == "📖 How It Works":
        _render_how_it_works()

    elif st.session_state.page == "🔬 Run Predictor":
        st.header("🔬 Malaria Occurrence Predictor")
        tab_predict, tab_batch = st.tabs(["Single Prediction", "Batch Prediction"])

        with tab_predict:
            if metadata is None:
                st.warning("No trained model found. Use the sidebar button to train models first.")
            else:
                payload = _render_prediction_form(defaults=metadata.get("feature_defaults", {}))
                if payload is not None:
                    try:
                        prediction = predict_dataframe(
                            input_df=payload,
                            model_dir=model_dir,
                            model_name=selected_model_name,
                            validation_config=validation_cfg,
                        ).iloc[0]
                        _render_prediction_output(result=prediction, threshold=threshold)
                    except ValueError as exc:
                        st.error(str(exc))

        with tab_batch:
            st.subheader("Batch Prediction from CSV")
            st.caption("Upload a CSV file containing clinical features to batch predict malaria occurrence.")
            uploaded_file = st.file_uploader("Upload Input CSV", type=["csv"])

            if uploaded_file is not None:
                batch_input = pd.read_csv(uploaded_file)
                st.write("Uploaded Dataset Preview:")
                st.dataframe(batch_input.head(20), use_container_width=True, hide_index=True)

                if st.button("Process Batch Predictions", use_container_width=True):
                    if metadata is None:
                        st.error("No trained model available yet. Train models from the sidebar first.")
                    else:
                        try:
                            batch_predictions = predict_dataframe(
                                input_df=batch_input,
                                model_dir=model_dir,
                                model_name=selected_model_name,
                                validation_config=validation_cfg,
                            )
                            st.success(f"Generated predictions for {len(batch_predictions)} records.")
                            st.dataframe(batch_predictions, use_container_width=True, hide_index=True)
                            csv_bytes = batch_predictions.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                "Download Predictions CSV",
                                data=csv_bytes,
                                file_name="predictions.csv",
                                mime="text/csv",
                                use_container_width=True,
                            )
                        except ValueError as exc:
                            st.error(str(exc))

    elif st.session_state.page == "📊 Model Insights":
        st.header("📊 Model Performance & Insights")
        if comparison_df is None:
            st.info("Model metrics are unavailable until at least one training run has completed.")
        else:
            _render_model_summary_cards(comparison_df)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)

            chart_df = comparison_df.melt(
                id_vars=["model"],
                value_vars=["accuracy", "precision", "recall", "f1_score"],
                var_name="metric",
                value_name="score",
            )
            fig = px.bar(
                chart_df,
                x="model",
                y="score",
                color="metric",
                barmode="group",
                title="Model Metric Comparison",
            )
            st.plotly_chart(fig, use_container_width=True)

        explainability_df = load_explainability_report(explainability_path)
        if explainability_df is not None and not explainability_df.empty:
            st.markdown("#### Permutation Feature Importance")
            st.dataframe(explainability_df, use_container_width=True, hide_index=True)
            top_features = explainability_df.head(15).iloc[::-1]
            fig = px.bar(
                top_features,
                x="importance_mean",
                y="feature",
                orientation="h",
                title="Top Feature Importance (Permutation)",
            )
            st.plotly_chart(fig, use_container_width=True)

    elif st.session_state.page == "⚙️ Database & Ops":
        st.header("⚙️ Database & Run Operations")
        latest_run = load_latest_experiment(experiments_index)
        if latest_run:
            st.markdown("#### Latest Training Run Metadata")
            st.json(latest_run)

        st.markdown("#### Raw Dataset Preview")
        if dataset_path.exists():
            preview_df = load_dataset_preview(dataset_path)
            st.dataframe(preview_df, use_container_width=True, hide_index=True)
            st.markdown("#### Feature Statistics (Describe)")
            st.dataframe(preview_df.describe(include="all").transpose(), use_container_width=True)
        else:
            st.error(f"Dataset file not found: {dataset_path}")


if __name__ == "__main__":
    main()