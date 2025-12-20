import streamlit as st
import numpy as np
import pandas as pd
import joblib
import torch
import tensorflow as tf

from pytorch_tabnet.tab_model import TabNetClassifier
from rtdl_revisiting_models import FTTransformer

if "probs" not in st.session_state:
    st.session_state.probs = None
    st.session_state.pred_class = None
    st.session_state.confidence = None


# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="Weather Classification",
    page_icon="🌦️",
    layout="wide",
)

# =========================
# Utility
# =========================
def safe_label_transform(le, series):
    known = set(le.classes_)
    return (
        series.astype(str)
        .apply(lambda x: x if x in known else le.classes_[0])
        .pipe(le.transform)
    )


def user_input(num_cols, cat_cols, csv_cat_values):
    num_data, cat_data = {}, {}

    with st.expander("🔢 Numeric Features", expanded=True):
        cols = st.columns(2)
        for i, col in enumerate(num_cols):
            num_data[col] = cols[i % 2].number_input(
                col, value=0.0, format="%.3f"
            )

    with st.expander("🔠 Categorical Features", expanded=True):
        cols = st.columns(2)
        for i, col in enumerate(cat_cols):
            cat_data[col] = cols[i % 2].selectbox(
                col,
                csv_cat_values[col],
            )

    return pd.DataFrame([{**num_data, **cat_data}])



# =========================
# Loaders
# =========================
@st.cache_data
def load_csv_metadata(csv_path, cat_cols):
    df = pd.read_csv(csv_path)

    cat_values = {}
    for col in cat_cols:
        cat_values[col] = sorted(df[col].astype(str).dropna().unique())

    return cat_values


@st.cache_resource
def load_mlp():
    return (
        tf.keras.models.load_model("artifacts/mlp/model.keras"),
        joblib.load("artifacts/mlp/preprocessor.joblib"),
        joblib.load("artifacts/mlp/label_encoder.joblib"),
        joblib.load("artifacts/mlp/num_cols.joblib"),
        joblib.load("artifacts/mlp/cat_cols.joblib"),
    )


@st.cache_resource
def load_tabnet():
    model = TabNetClassifier()
    model.load_model("artifacts/tabnet/tabnet_model.zip")
    return (
        model,
        joblib.load("artifacts/tabnet/cat_encoders.joblib"),
        joblib.load("artifacts/tabnet/num_cols.joblib"),
        joblib.load("artifacts/tabnet/cat_cols.joblib"),
        joblib.load("artifacts/tabnet/label_encoder.joblib"),
    )


@st.cache_resource
def load_ft():
    checkpoint = torch.load(
        "artifacts/ft_transformer/model.pt",
        map_location="cpu",
        weights_only=False,
    )

    model = FTTransformer(
        n_cont_features=len(checkpoint["num_cols"]),
        cat_cardinalities=[
            len(checkpoint["cat_encoders"][c].classes_)
            for c in checkpoint["cat_cols"]
        ],
        d_block=32,
        n_blocks=4,
        attention_n_heads=8,
        attention_dropout=0.1,
        ffn_d_hidden_multiplier=4 / 3,
        ffn_dropout=0.1,
        residual_dropout=0.1,
        d_out=checkpoint["num_classes"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return (
        model,
        checkpoint["cat_encoders"],
        checkpoint["scaler"],
        checkpoint["label_encoder"],
        checkpoint["num_cols"],
        checkpoint["cat_cols"],
    )


# =========================
# Sidebar
# =========================
st.sidebar.title("⚙️ Configuration")

model_choice = st.sidebar.radio(
    "Select Model",
    ["MLP", "TabNet", "FT-Transformer"],
)

if st.session_state.get("last_model") != model_choice:
    st.session_state.probs = None
    st.session_state.last_model = model_choice

st.sidebar.markdown("---")
st.sidebar.info(
    """
**Model Details**
- **MLP**: Neural Network + ColumnTransformer  
- **TabNet**: Attention-based tabular DL  
- **FT-Transformer**: Transformer for tabular data
"""
)

# =========================
# Main UI
# =========================
st.title("🌦️ Weather Classification Dashboard")
st.caption("Compare deep learning models for tabular weather data")

# =========================
# Model Selection
# =========================
if model_choice == "MLP":
    model, preprocessor, le, num_cols, cat_cols = load_mlp()

elif model_choice == "TabNet":
    model, cat_encoders, num_cols, cat_cols, le = load_tabnet()

else:
    model, cat_encoders, scaler, le, num_cols, cat_cols = load_ft()

CSV_PATH = "data/weather_classification_data.csv"
csv_cat_values = load_csv_metadata(CSV_PATH, cat_cols)

# =========================
# Input
# =========================
input_df = user_input(num_cols, cat_cols, csv_cat_values)

# =========================
# Predict Button
# =========================
if st.button("🔍 Predict", width="stretch"):
    if model_choice == "MLP":
        X = preprocessor.transform(input_df)
        probs = model.predict(X)[0]

    elif model_choice == "TabNet":
        for col in cat_cols:
            input_df[col] = safe_label_transform(cat_encoders[col], input_df[col])
        X = input_df[num_cols + cat_cols].values
        probs = model.predict_proba(X)[0]

    else:
        for col in cat_cols:
            input_df[col] = safe_label_transform(cat_encoders[col], input_df[col])

        X_num = torch.tensor(
            scaler.transform(input_df[num_cols]),
            dtype=torch.float32,
        )
        X_cat = torch.tensor(
            input_df[cat_cols].values,
            dtype=torch.long,
        )

        with torch.no_grad():
            logits = model(X_num, X_cat)
            probs = torch.softmax(logits, dim=1).numpy()[0]

    pred_idx = np.argmax(probs)

    st.session_state.probs = probs
    st.session_state.pred_class = le.inverse_transform([pred_idx])[0]
    st.session_state.confidence = probs[pred_idx]


if st.session_state.probs is not None:
    st.markdown("## 🧠 Prediction Result")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.metric(
            label="Predicted Class",
            value=st.session_state.pred_class,
            delta=f"{st.session_state.confidence*100:.2f}% confidence",
        )

    with col2:
        prob_df = pd.DataFrame(
            {
                "Class": le.classes_,
                "Probability": st.session_state.probs,
            }
        ).sort_values("Probability", ascending=False)

        st.dataframe(
            prob_df,
            width="stretch",
            hide_index=True,
        )

    st.markdown("### 📊 Class Probability Distribution")
    st.bar_chart(
        prob_df.set_index("Class")["Probability"],
        height=300,
    )


