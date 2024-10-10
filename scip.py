import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from PIL import Image

# Charger les modèles et le scaler
model_solvabilite = joblib.load('model_solvabilite.pkl')
model_credit_score = joblib.load('model_credit_score.pkl')
model_pret = joblib.load('model_pret.pkl')
scaler = joblib.load('scaler.pkl')

# Créer de nouveaux encodeurs pour les variables catégorielles
encoders = {}
for column in ['job', 'marital', 'education', 'housing', 'loan']:
    encoders[column] = LabelEncoder()

# Titre principal de l'application avec une taille de caractère ajustée
st.markdown("<h1 style='text-align: center; color: #2C3E50;'>Prédiction de Solvabilité et du Prêt</h1>", unsafe_allow_html=True)

# Définir le style général des textes et des bordures pour le formulaire
st.markdown("""
    <style>
    .big-font { font-size:20px !important; color: #34495E; }
    .medium-font { font-size:20px !important; color: #2ECC71; }
    .small-font { font-size:18px !important; color: #7F8C8D; }
    
    /* Bordure et style des inputs */
    .stNumberInput, .stSelectbox {
        border: 2px solid #2ECC71;
        padding: 10px;
        margin-bottom: 15px;
        border-radius: 5px;
    }

    /* Style des labels */
    label {
        font-size: 18px;
        color: #2980B9;
        font-weight: bold;
    }
    
    /* Bordure des colonnes pour mieux visualiser */
    .stColumn {
        border: 2px solid #2980B9;
        border-radius: 10px;
        padding: 20px;
    }

    /* Centrer les icônes et textes */
    .center {
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Créer deux colonnes pour structurer l'interface utilisateur avec bordure
col1, col2 = st.columns(2)

# ---------------- Formulaire 1 : Prédire la Solvabilité et le Score de Crédit ----------------
with col1:
    

    # Formulaire pour les données utilisateurs
    age = st.number_input('Âge', min_value=18, max_value=120, value=30, key='age_input')
    job = st.selectbox("Profession", options=["admin.", "technician", "services", "management", "retired", "blue-collar", "unemployed", "entrepreneur", "housemaid", "self-employed", "student"], key='job_input')
    marital = st.selectbox("État civil", options=["married", "single", "divorced"], key='marital_input')
    education = st.selectbox("Niveau d'éducation", options=["primary", "secondary", "tertiary", "unknown"], key='education_input')
    balance = st.number_input("Solde bancaire($)", min_value=-100000, max_value=1000000, value=0, key='balance_input')
    housing = st.selectbox("Possession d'une maison", options=["yes", "no"], key='housing_input')
    loan = st.selectbox("Prêt en cours", options=["yes", "no"], key='loan_input')

    # Encodage des variables catégorielles
    input_data = {
        'age': [age],
        'job': [encoders['job'].fit_transform([job])[0]],  # Encodage dynamique
        'marital': [encoders['marital'].fit_transform([marital])[0]],
        'education': [encoders['education'].fit_transform([education])[0]],
        'balance': [balance],
        'housing': [encoders['housing'].fit_transform([housing])[0]],
        'loan': [encoders['loan'].fit_transform([loan])[0]]
    }
    
    df_input = pd.DataFrame(input_data)
    df_input_scaled = scaler.transform(df_input)

    if st.button('Prédire Solvabilité'):
        solvabilite_pred = model_solvabilite.predict(df_input_scaled)
        solvable = solvabilite_pred[0] == 1  # Vérifier si le client est solvable

        if solvable:
            st.markdown("<p class='medium-font center'>✅ Solvabilité : L'emprunteur est Solvable.</p>", unsafe_allow_html=True)
            # Prédire le score de crédit
            score_credit_pred = model_credit_score.predict(df_input)
            st.markdown(f"<p class='medium-font center'>📊 Score de Crédit : {score_credit_pred[0]:.2f}</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='medium-font center'>❌ L'emprunteur n'est pas solvable, aucun prêt ne peut être accordé.</p>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)  # Fin de la bordure de la colonne

# ---------------- Formulaire 2 : Prédire le Montant du Prêt ----------------
with col2:
    
    # Formulaire pour saisir le score de crédit
    score_credit_input = st.number_input('Entrez le Score de Crédit ', min_value=0.0, max_value=1000.0, value=0.0, key='score_credit_input')

    if st.button('Prédire Montant du Prêt'):
        df_pret = pd.DataFrame({'credit_score': [score_credit_input]})
        montant_pret_pred = model_pret.predict(df_pret)
        st.markdown(f"<p class='medium-font center'>💰 Montant du Prêt Prédit : {montant_pret_pred[0]:,.2f}$</p>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)  # Fin de la bordure de la colonne

# Ajouter un footer avec des informations supplémentaires
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p class='small-font center'>© 2024 - Prédiction de Solvabilité & Crédit</p>", unsafe_allow_html=True)
