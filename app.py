import streamlit as st
import pandas as pd
import pickle
from sdv.single_table import CTGANSynthesizer

# ---------------------
# Page configuration
# ---------------------
st.set_page_config(page_title="CTGAN Synthetic Data Generator", layout="wide")

st.title("🤖 CTGAN Synthetic Data Generator")
st.markdown("""
Generate synthetic tabular data with CTGAN and compare it to real data.
Upload your model (.pkl) and/or real dataset (.csv).
""")

# ---------------------
# Sidebar controls
# ---------------------
st.sidebar.header("⚙️ Controls")
uploaded_model = st.sidebar.file_uploader("Upload trained CTGAN model (.pkl)", type="pkl")
uploaded_real = st.sidebar.file_uploader("Upload real dataset (.csv)", type="csv")

num_samples = st.sidebar.slider(
    "Number of synthetic samples", min_value=10, max_value=5000, value=100, step=10
)

show_preview = st.sidebar.checkbox("Show synthetic preview", value=True)
generate_button = st.sidebar.button("Generate Synthetic Data")

# ---------------------
# Load CTGAN model
# ---------------------
@st.cache_resource
def load_model(file):
    return pickle.load(file)

ctgan = None
if uploaded_model is not None:
    try:
        ctgan = load_model(uploaded_model)
        st.success("✅ Model loaded successfully")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")

# ---------------------
# Load real dataset
# ---------------------
@st.cache_data
def load_real_data(file):
    return pd.read_csv(file)

real_data = None
if uploaded_real is not None:
    try:
        real_data = load_real_data(uploaded_real)
        st.success("✅ Real dataset loaded successfully")
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")

# ---------------------
# Generate synthetic data
# ---------------------
synthetic_data = None
if generate_button and ctgan is not None:
    with st.spinner("Generating synthetic data... ⏳"):
        try:
            synthetic_data = ctgan.sample(num_samples)
        except Exception as e:
            st.error(f"❌ Error during sampling: {e}")

    if synthetic_data is not None:
        st.success(f"✅ Synthetic data ready — {synthetic_data.shape[0]} rows")

        # Preview synthetic data
        if show_preview:
            st.subheader("📊 Synthetic Data Preview")
            preview_rows = st.sidebar.slider(
                "Rows to preview",
                min_value=5,
                max_value=min(50, num_samples),
                value=min(10, num_samples),
                step=5,
            )
            st.dataframe(synthetic_data.head(preview_rows))

        # Download button
        st.subheader("📥 Download Synthetic Data")
        csv_data = synthetic_data.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name="synthetic_data.csv",
            mime="text/csv",
        )
