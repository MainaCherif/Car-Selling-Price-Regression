"""
App Streamlit - Prédiction du prix de vente d'une voiture (Car_data, régression)

Fichiers attendus dans le MÊME dossier que ce script (générés par le notebook
Reg_Random_Forest_XGBoost.ipynb, section "Tester le modèle le plus performant") :
    - best_model.joblib
    - scaler.joblib
    - encoders.joblib
    - uniques.joblib

Lancer en local :
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib as jb
import os

st.set_page_config(page_title="Prédiction prix de vente voiture", page_icon="🚙")

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_artifacts():
    model = jb.load(os.path.join(MODEL_DIR, "best_model.joblib"))
    scaler = jb.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    encoders = jb.load(os.path.join(MODEL_DIR, "encoders.joblib"))
    uniques = jb.load(os.path.join(MODEL_DIR, "uniques.joblib"))
    return model, scaler, encoders, uniques


try:
    model, scaler, encoders, uniques = load_artifacts()
except Exception as e:
    st.error(
        "Fichiers .joblib introuvables dans le dossier de l'app. "
        "Copie best_model.joblib, scaler.joblib, encoders.joblib et uniques.joblib "
        "à côté de app.py, puis relance.\n\nDétail : " + str(e)
    )
    st.stop()

# uniques / encoders : [Fuel_Type, Seller_Type, Transmission]
fuel_types = uniques[0]
seller_types = uniques[1]
transmissions = uniques[2]

FEATURE_ORDER = ["Kms_Driven", "Present_Price", "Fuel_Type", "Seller_Type", "Transmission", "Age"]

st.title("🚙 Prédire le prix de vente d'une voiture d'occasion")
st.caption("Modèle entrainé sur un dataset de voitures d'occasion (kilométrage, prix catalogue, âge, etc.).")

tab1, tab2 = st.tabs(["Prédiction simple", "Prédiction par fichier CSV"])

# ---------------------------------------------------------------------------
# Prédiction simple
# ---------------------------------------------------------------------------
with tab1:
    with st.form("form_simple"):
        col1, col2 = st.columns(2)
        with col1:
            kms = st.number_input("Kilométrage parcouru (km)", min_value=0, value=30000, step=1000)
            present_price = st.number_input("Prix catalogue actuel (en lakhs, ex: 5.59)", min_value=0.0, value=6.0, step=0.1)
            fuel = st.selectbox("Type de carburant", fuel_types)
        with col2:
            seller = st.selectbox("Type de vendeur", seller_types)
            transmission = st.selectbox("Transmission", transmissions)
            age = st.number_input("Âge de la voiture (années)", min_value=0, max_value=30, value=5, step=1)

        submitted = st.form_submit_button("Prédire")

    if submitted:
        fuel_e = encoders[0].transform([fuel])[0]
        seller_e = encoders[1].transform([seller])[0]
        transmission_e = encoders[2].transform([transmission])[0]

        # même ordre que dans le notebook : Kms_Driven, Present_Price, Fuel_Type, Seller_Type, Transmission, Age
        x_new = np.array([[kms, present_price, fuel_e, seller_e, transmission_e, age]])
        x_new = scaler.transform(x_new)

        y_pred = model.predict(x_new)[0]
        y_pred = max(0, float(y_pred))

        st.success(f"Prix de vente estimé : **{y_pred:.2f} lakhs**")

# ---------------------------------------------------------------------------
# Prédiction par lot (CSV)
# ---------------------------------------------------------------------------
with tab2:
    st.write("Le fichier doit contenir les colonnes suivantes, dans cet ordre : `" + ", ".join(FEATURE_ORDER) + "`.")
    uploaded = st.file_uploader("Importer un fichier CSV", type=["csv"])

    if uploaded is not None:
        df_in = pd.read_csv(uploaded)
        try:
            df_enc = df_in.copy()
            df_enc["Fuel_Type"] = encoders[0].transform(df_enc["Fuel_Type"])
            df_enc["Seller_Type"] = encoders[1].transform(df_enc["Seller_Type"])
            df_enc["Transmission"] = encoders[2].transform(df_enc["Transmission"])

            x_batch = df_enc[FEATURE_ORDER].values
            x_batch = scaler.transform(x_batch)
            preds = model.predict(x_batch)
            preds = np.clip(preds, 0, None)

            df_out = df_in.copy()
            df_out["Selling_Price_predit"] = preds
            st.dataframe(df_out)

            csv_bytes = df_out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Télécharger les prédictions (CSV)",
                data=csv_bytes,
                file_name="predictions_car_data_selling_price.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Erreur pendant le traitement du fichier : {e}")
