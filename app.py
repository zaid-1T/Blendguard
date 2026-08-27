# app.py

import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf


# =========================================================
# FILE PATHS
# =========================================================

MODEL_PATH = "models/ann/blendguard_ann_final.keras"
X_SCALER_PATH = "models/ann/scalers/X_scaler.joblib"
Y_SCALER_PATH = "models/ann/scalers/y_scaler.joblib"
METADATA_PATH = "data/ann/ann_metadata.json"


# =========================================================
# SPECIFICATION LIMITS
# =========================================================

SPEC_LIMITS = {
    "AKI_min": 87.0,
    "RVP_max": 15.0,
    "RVP_1_25_max": 29.5,
    "benzene_max": 1.1
}


# =========================================================
# PAGE SETUP
# =========================================================

st.set_page_config(
    page_title="BlendGuard AI",
    page_icon="⛽",
    layout="wide"
)


# =========================================================
# STYLING
# =========================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #666666;
        margin-top: 0px;
        margin-bottom: 24px;
    }

    .pass-box {
        background-color: #e8f7ee;
        border: 1px solid #8fd19e;
        color: #14532d;
        padding: 22px;
        border-radius: 14px;
        font-size: 28px;
        font-weight: 800;
        text-align: center;
    }

    .fail-box {
        background-color: #fdecec;
        border: 1px solid #f5a3a3;
        color: #7f1d1d;
        padding: 22px;
        border-radius: 14px;
        font-size: 28px;
        font-weight: 800;
        text-align: center;
    }

    .info-box {
        background-color: #eef4ff;
        border: 1px solid #adc8ff;
        color: #1e3a8a;
        padding: 16px;
        border-radius: 12px;
        font-size: 16px;
    }

    .warning-box {
        background-color: #fff7e6;
        border: 1px solid #f4c430;
        color: #7a4b00;
        padding: 16px;
        border-radius: 12px;
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# LOAD TRAINED MODEL AND SCALERS
# =========================================================

@st.cache_resource
def load_backend():
    required_files = [
        MODEL_PATH,
        X_SCALER_PATH,
        Y_SCALER_PATH,
        METADATA_PATH
    ]

    missing_files = [
        file_path for file_path in required_files
        if not os.path.exists(file_path)
    ]

    if missing_files:
        raise FileNotFoundError(
            "Missing required files:\n" + "\n".join(missing_files)
        )

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    x_scaler = joblib.load(X_SCER_PATH) if False else joblib.load(X_SCALER_PATH)
    y_scaler = joblib.load(Y_SCALER_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    feature_columns = metadata["feature_columns"]

    target_columns = metadata.get("ann_target_columns")
    if target_columns is None:
        target_columns = metadata.get("target_columns")

    if target_columns is None:
        raise KeyError(
            "Could not find ann_target_columns or target_columns in ann_metadata.json"
        )

    expected_targets = [
        "RON_nonlinear",
        "MON_nonlinear",
        "RVP_1_25",
        "benzene",
        "cost_per_gal"
    ]

    if target_columns != expected_targets:
        raise ValueError(
            f"Wrong target columns found: {target_columns}. "
            f"Expected: {expected_targets}"
        )

    return model, x_scaler, y_scaler, feature_columns, target_columns


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def clean_name(column_name):
    name = column_name.replace("_pct", "")
    name = name.replace("_", " ")
    return name.title()


def predict_properties(
    recipe_fractions,
    model,
    x_scaler,
    y_scaler,
    feature_columns,
    target_columns
):
    input_df = pd.DataFrame(
        [recipe_fractions],
        columns=feature_columns
    )

    input_scaled = x_scaler.transform(input_df)

    prediction_scaled = model.predict(
        input_scaled,
        verbose=0
    )

    prediction_raw = y_scaler.inverse_transform(
        prediction_scaled
    )

    prediction_df = pd.DataFrame(
        prediction_raw,
        columns=target_columns
    )

    prediction = prediction_df.iloc[0].to_dict()

    prediction["AKI_nonlinear"] = (
        prediction["RON_nonlinear"]
        + prediction["MON_nonlinear"]
    ) / 2

    if prediction["RVP_1_25"] >= 0:
        prediction["RVP"] = prediction["RVP_1_25"] ** 0.8
    else:
        prediction["RVP"] = np.nan

    return prediction


def calculate_pass_fail(prediction):
    if pd.isna(prediction["RVP"]):
        return False

    return (
        prediction["AKI_nonlinear"] >= SPEC_LIMITS["AKI_min"]
        and prediction["RVP"] <= SPEC_LIMITS["RVP_max"]
        and prediction["RVP_1_25"] <= SPEC_LIMITS["RVP_1_25_max"]
        and prediction["benzene"] <= SPEC_LIMITS["benzene_max"]
    )


def get_failure_reasons(prediction):
    reasons = []

    if prediction["AKI_nonlinear"] < SPEC_LIMITS["AKI_min"]:
        reasons.append("AKI too low")

    if pd.isna(prediction["RVP"]):
        reasons.append("RVP invalid")
    elif prediction["RVP"] > SPEC_LIMITS["RVP_max"]:
        reasons.append("RVP too high")

    if prediction["RVP_1_25"] > SPEC_LIMITS["RVP_1_25_max"]:
        reasons.append("RVP_1_25 too high")

    if prediction["benzene"] > SPEC_LIMITS["benzene_max"]:
        reasons.append("Benzene too high")

    if not reasons:
        reasons.append("PASS")

    return reasons


def calculate_margins(prediction):
    return {
        "AKI margin": prediction["AKI_nonlinear"] - SPEC_LIMITS["AKI_min"],
        "RVP margin": SPEC_LIMITS["RVP_max"] - prediction["RVP"],
        "RVP_1_25 margin": SPEC_LIMITS["RVP_1_25_max"] - prediction["RVP_1_25"],
        "Benzene margin": SPEC_LIMITS["benzene_max"] - prediction["benzene"]
    }


def get_suggestions(failure_reasons):
    suggestions = []

    if "AKI too low" in failure_reasons:
        suggestions.append(
            "Increase high-octane components such as alkylate or high-octane reformate, or reduce low-octane straight-run naphtha."
        )

    if "RVP too high" in failure_reasons or "RVP_1_25 too high" in failure_reasons:
        suggestions.append(
            "Reduce high-volatility components such as butane, or increase lower-volatility blending streams."
        )

    if "Benzene too high" in failure_reasons:
        suggestions.append(
            "Reduce benzene-heavy reformate streams or shift toward lower-benzene components."
        )

    if not suggestions:
        suggestions.append(
            "Recipe satisfies all current PDF-style regular gasoline constraints."
        )

    return suggestions


def fmt(value, decimals=3):
    if pd.isna(value):
        return "Invalid"
    return f"{value:.{decimals}f}"


# =========================================================
# APP BODY
# =========================================================

st.markdown(
    '<div class="main-title">BlendGuard AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">ANN-based gasoline blend property prediction and specification checking</div>',
    unsafe_allow_html=True
)


try:
    model, x_scaler, y_scaler, feature_columns, target_columns = load_backend()
except Exception as error:
    st.error("Could not load model, scalers, or metadata.")
    st.exception(error)
    st.stop()


st.sidebar.title("BlendGuard AI")
st.sidebar.write("Enter component percentages for one gasoline blend recipe.")

normalize_automatically = st.sidebar.checkbox(
    "Normalize recipe automatically",
    value=True
)

st.sidebar.markdown("---")
st.sidebar.write("Specification limits")
st.sidebar.write(f"AKI ≥ {SPEC_LIMITS['AKI_min']}")
st.sidebar.write(f"RVP ≤ {SPEC_LIMITS['RVP_max']}")
st.sidebar.write(f"RVP_1_25 ≤ {SPEC_LIMITS['RVP_1_25_max']}")
st.sidebar.write(f"Benzene ≤ {SPEC_LIMITS['benzene_max']} vol%")


st.subheader("1. Enter Blend Recipe")

st.write(
    "Enter component values as volume percentages. "
    "The recipe should ideally add to 100%."
)


default_values = {
    "butane_pct": 8.0,
    "straight_run_naphtha_pct": 12.0,
    "isomerate_pct": 10.0,
    "reformate_high_octane_pct": 20.0,
    "reformate_low_benzene_pct": 5.0,
    "fcc_naphtha_pct": 30.0,
    "alkylate_pct": 15.0
}

input_percentages = {}

cols = st.columns(2)

for index, feature in enumerate(feature_columns):
    with cols[index % 2]:
        input_percentages[feature] = st.number_input(
            clean_name(feature),
            min_value=0.0,
            max_value=100.0,
            value=float(default_values.get(feature, 0.0)),
            step=0.5,
            format="%.2f"
        )

total_percentage = sum(input_percentages.values())

st.metric(
    "Recipe total",
    f"{total_percentage:.2f}%"
)

if abs(total_percentage - 100.0) > 0.001:
    if normalize_automatically:
        st.markdown(
            """
            <div class="warning-box">
            Recipe does not add to 100%. The app will normalize the values before prediction.
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.error("Recipe must add to 100%, or enable automatic normalization.")
        st.stop()


if st.button("Predict Blend Properties", type="primary", use_container_width=True):

    if total_percentage <= 0:
        st.error("Total recipe percentage must be greater than zero.")
        st.stop()

    if normalize_automatically:
        recipe_fractions = {
            feature: input_percentages[feature] / total_percentage
            for feature in feature_columns
        }
    else:
        recipe_fractions = {
            feature: input_percentages[feature] / 100.0
            for feature in feature_columns
        }

    prediction = predict_properties(
        recipe_fractions=recipe_fractions,
        model=model,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        feature_columns=feature_columns,
        target_columns=target_columns
    )

    passes = calculate_pass_fail(prediction)
    failure_reasons = get_failure_reasons(prediction)
    margins = calculate_margins(prediction)
    suggestions = get_suggestions(failure_reasons)

    st.markdown("---")
    st.subheader("2. Specification Result")

    if passes:
        st.markdown(
            '<div class="pass-box">PASS</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="fail-box">FAIL</div>',
            unsafe_allow_html=True
        )

    st.subheader("3. Predicted Fuel Properties")

    metric_cols = st.columns(4)

    metric_cols[0].metric("RON", fmt(prediction["RON_nonlinear"], 3))
    metric_cols[1].metric("MON", fmt(prediction["MON_nonlinear"], 3))
    metric_cols[2].metric("AKI", fmt(prediction["AKI_nonlinear"], 3))
    metric_cols[3].metric("RVP", fmt(prediction["RVP"], 3))

    metric_cols_2 = st.columns(3)

    metric_cols_2[0].metric("RVP_1_25", fmt(prediction["RVP_1_25"], 3))
    metric_cols_2[1].metric("Benzene vol%", fmt(prediction["benzene"], 4))
    metric_cols_2[2].metric("Cost per gal", fmt(prediction["cost_per_gal"], 4))

    st.subheader("4. Failure Reasons")

    if passes:
        st.success("The blend satisfies all current PDF-style regular gasoline constraints.")
    else:
        for reason in failure_reasons:
            st.error(reason)

    st.subheader("5. Specification Margins")

    margin_df = pd.DataFrame(
        [
            {
                "Constraint": "AKI ≥ 87",
                "Margin": margins["AKI margin"],
                "Meaning": "Positive means AKI is high enough"
            },
            {
                "Constraint": "RVP ≤ 15",
                "Margin": margins["RVP margin"],
                "Meaning": "Positive means RVP is below the limit"
            },
            {
                "Constraint": "RVP_1_25 ≤ 29.5",
                "Margin": margins["RVP_1_25 margin"],
                "Meaning": "Positive means RVP_1_25 is below the limit"
            },
            {
                "Constraint": "Benzene ≤ 1.1 vol%",
                "Margin": margins["Benzene margin"],
                "Meaning": "Positive means benzene is below the limit"
            }
        ]
    )

    st.dataframe(
        margin_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("6. Recipe Used by Model")

    recipe_df = pd.DataFrame(
        [
            {
                "Component": clean_name(feature),
                "Entered percentage": input_percentages[feature],
                "Model fraction": recipe_fractions[feature]
            }
            for feature in feature_columns
        ]
    )

    st.dataframe(
        recipe_df,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("7. Suggested Adjustment")

    for suggestion in suggestions:
        st.write(f"- {suggestion}")

    st.markdown("---")

    st.markdown(
        """
        <div class="info-box">
        Methodology note: The ANN predicts five independent properties: RON, MON,
        RVP_1_25, benzene and cost. AKI and RVP are then derived using known equations.
        The final PASS/FAIL decision is made using specification rules. The ANN is not
        directly trained as a pass/fail classifier.
        </div>
        """,
        unsafe_allow_html=True
    )