# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import matplotlib.pyplot as plt

st.set_page_config(page_title="Salary Predictor", layout="centered")

@st.cache_resource
def load_model(path="salary_prediction_model.pkl"):
    try:
        model = joblib.load(path)
    except Exception as e:
        st.error(f"Could not load model file at '{path}': {e}")
        return None
    return model

model = load_model("salary_prediction_model.pkl")

st.title("📈 Salary Prediction (Linear Regression)")
st.caption("Predict salary using features: Age, Experience, EducationLevel")

if model is None:
    st.warning("Model not loaded. Make sure `salary_prediction_model.pkl` is in this folder.")
    st.stop()

# ---- Single prediction ----
st.header("Single prediction")

with st.form("single_predict"):
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
    with col2:
        experience = st.number_input("Experience (years)", min_value=0, max_value=80, value=8, step=1)
    with col3:
        edu_choice = st.selectbox("Education Level (years)", options=[10, 12, 14, 16, 18], index=3)
    submitted = st.form_submit_button("Predict")

if submitted:
    X_single = pd.DataFrame({
        "Age": [age],
        "Experience": [experience],
        "EducationLevel": [edu_choice]
    })
    pred = model.predict(X_single)[0]
    st.success(f"Predicted Salary: ₹ {pred:,.2f}")
    # show model internals
    try:
        coefs = model.coef_
        intercept = model.intercept_
        feat_names = ["Age", "Experience", "EducationLevel"]
        st.write("Model intercept:", float(intercept))
        cof_df = pd.DataFrame({"Feature": feat_names, "Coefficient": np.round(coefs, 3)})
        st.table(cof_df)
    except Exception:
        st.info("Model does not expose coefficients (might be a pipeline or non-linear model).")

st.markdown("---")

# ---- Batch prediction (CSV) ----
st.header("Batch prediction (CSV)")
st.markdown("Upload a CSV with columns: `Age`, `Experience`, `EducationLevel`. You can include `Salary` to compare actual vs predicted.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error("Failed to read CSV: " + str(e))
        st.stop()

    required_cols = {"Age", "Experience", "EducationLevel"}
    if not required_cols.issubset(set(data.columns)):
        st.error(f"CSV must contain columns: {required_cols}. Found: {list(data.columns)}")
    else:
        X_batch = data[["Age", "Experience", "EducationLevel"]].copy()
        preds = model.predict(X_batch)
        data["Predicted_Salary"] = preds
        st.success(f"Predicted {len(preds)} rows")
        st.dataframe(data.head(50))

        # allow download
        csv_buf = io.StringIO()
        data.to_csv(csv_buf, index=False)
        csv_bytes = csv_buf.getvalue().encode()
        st.download_button(
            label="Download predictions as CSV",
            data=csv_bytes,
            file_name="predictions.csv",
            mime="text/csv"
        )

        # If actual Salary present, show scatter plot Actual vs Predicted
        if "Salary" in data.columns:
            st.subheader("Actual vs Predicted (uploaded CSV)")
            fig, ax = plt.subplots(figsize=(6,4))
            ax.scatter(data["Salary"], data["Predicted_Salary"], alpha=0.6)
            ax.plot([data["Salary"].min(), data["Salary"].max()],
                    [data["Salary"].min(), data["Salary"].max()],
                    linestyle="--")
            ax.set_xlabel("Actual Salary")
            ax.set_ylabel("Predicted Salary")
            ax.set_title("Actual vs Predicted")
            st.pyplot(fig)

st.markdown("---")

# ---- Utilities ----
st.header("Utilities")
st.write("If you don't have a sample CSV, download one to test the batch prediction.")
if st.button("Create sample CSV (100 rows)"):
    sample = pd.DataFrame({
        "Age": np.random.randint(18, 65, 100),
    })
    sample["Experience"] = (sample["Age"] - 18 + np.random.randint(-2,5,size=100)).clip(0)
    sample["EducationLevel"] = np.random.choice([10,12,14,16,18], size=100)
    buf = io.StringIO()
    sample.to_csv(buf, index=False)
    st.download_button("Download sample CSV", data=buf.getvalue().encode(), file_name="sample_input.csv", mime="text/csv")

st.write("Run this app with:")
st.code("streamlit run app.py", language="bash")
st.caption("Make sure `salary_prediction_model.pkl` is in the same folder as this script.")
