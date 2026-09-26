"""Interactive client. Run: streamlit run streamlit_app.py"""

import math
import os

import requests
import streamlit as st

st.set_page_config(page_title="Loan Approval Predictor", page_icon="🏦", layout="centered")
API_URL = os.environ.get("LOAN_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.title("Loan Approval Predictor")
st.caption("Educational model trained on a public Kaggle dataset. Actual customer provenance is unverified.")
st.info("Use the dataset's original monetary and loan-term units. The source does not document their units.")

with st.form("applicant_form"):
    left, right = st.columns(2)
    with left:
        cibil_score = st.number_input("CIBIL score", min_value=300, max_value=900, value=750, step=1)
        income_annum = st.number_input("Annual income", min_value=1.0, value=2400000.0, step=100000.0, format="%.0f")
        loan_amount = st.number_input("Requested loan amount", min_value=1.0, value=6000000.0, step=100000.0, format="%.0f")
        loan_term = st.number_input("Loan term (dataset units)", min_value=1, value=10, step=1,
                                    help="The training data contains terms from 2 to 20.")
        no_of_dependents = st.number_input("Number of dependents", min_value=0, value=2, step=1)
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    with right:
        self_employed = st.selectbox("Self-employed", ["No", "Yes"])
        residential_assets_value = st.number_input("Residential assets value", min_value=0.0, value=5000000.0, step=100000.0, format="%.0f")
        commercial_assets_value = st.number_input("Commercial assets value", min_value=0.0, value=0.0, step=100000.0, format="%.0f")
        luxury_assets_value = st.number_input("Luxury assets value", min_value=0.0, value=3000000.0, step=100000.0, format="%.0f")
        bank_asset_value = st.number_input("Bank assets value", min_value=0.0, value=2000000.0, step=100000.0, format="%.0f")
    submitted = st.form_submit_button("Predict loan decision", type="primary")

if submitted:
    payload = {
        "no_of_dependents": int(no_of_dependents), "education": education,
        "self_employed": self_employed, "income_annum": float(income_annum),
        "loan_amount": float(loan_amount), "loan_term": int(loan_term),
        "cibil_score": int(cibil_score),
        "residential_assets_value": float(residential_assets_value),
        "commercial_assets_value": float(commercial_assets_value),
        "luxury_assets_value": float(luxury_assets_value),
        "bank_asset_value": float(bank_asset_value),
    }
    try:
        with st.spinner("Getting prediction..."):
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=(5, 20))
        if response.status_code == 422:
            errors = response.json()["detail"]
            messages = [f"{'.'.join(map(str, item['loc'][1:]))}: {item['msg']}" for item in errors]
            st.error("Please check your inputs. " + "; ".join(messages))
        elif response.status_code == 503:
            st.error("The model is unavailable. Run the training and API cells, then try again.")
        else:
            response.raise_for_status()
            result = response.json()
            score = float(result["approval_score"])
            model_name = result["model_name"]
            if result["prediction"] not in {"Approved", "Rejected"} or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Unexpected prediction response")
            if result["prediction"] == "Approved":
                st.success("Predicted decision: Approved")
            else:
                st.warning("Predicted decision: Rejected")
            st.metric("Model approval score", f"{score:.1%}")
            st.progress(score)
            st.caption("This is an uncalibrated model score. A lender determines the actual loan decision.")
            st.caption(f"Model: {model_name}")
    except requests.Timeout:
        st.error("The API timed out. Check that the Colab runtime is connected and try again.")
    except requests.ConnectionError:
        st.error("Cannot reach the API. Run the FastAPI server cell first.")
    except requests.HTTPError as exc:
        st.error(f"The API returned HTTP {exc.response.status_code}. Check the API server log.")
    except requests.RequestException:
        st.error("The API request failed. Check the backend URL and server.")
    except (ValueError, KeyError, TypeError):
        st.error("The API returned an invalid response. Check that the matching backend is running.")
