import streamlit as st
from transformers import pipeline
from PIL import Image
import pandas as pd
import os

st.set_page_config(page_title="Analyse IA Multimodale", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à :", ["Analyse d'Images", "Analyse de Texte", "Tableau de Bord"])

# Charger modèles IA
image_classifier = pipeline("image-classification")
sentiment_analyzer = pipeline("sentiment-analysis")

# Créer un historique s'il n'existe pas
if not os.path.exists("historique.csv"):
    pd.DataFrame(columns=["type", "input", "result"]).to_csv("historique.csv", index=False)

historique = pd.read_csv("historique.csv")

#PAGE 1 : ANALYSE D'IMAGES 
if page == "Analyse d'Images":
    st.title("🔍 Analyse d'Images avec IA")

    uploaded_files = st.file_uploader("Choisis une ou plusieurs images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Image : {uploaded_file.name}", width=300)

            st.write("Analyse en cours...")
            results = image_classifier(image)

            st.write("Résultats :")
            for result in results:
                st.write(f"- **{result['label']}** ({result['score']:.2f})")

            # sauver dans historique
            historique.loc[len(historique)] = ["image", uploaded_file.name, results[0]["label"]]
            historique.to_csv("historique.csv", index=False)

# PAGE 2 : ANALYSE DE TEXTE 
elif page == "Analyse de Texte":
    st.title("📝 Analyse de Texte avec IA")

    user_text = st.text_area("Écris un texte")

    if st.button("Analyser"):
        result = sentiment_analyzer(user_text)[0]
        st.write(f"Sentiment : **{result['label']}** (score : {result['score']:.2f})")

        historique.loc[len(historique)] = ["texte", user_text, result["label"]]
        historique.to_csv("historique.csv", index=False)

#PAGE 3 : TABLEAU DE BORD
else:
    st.title("📊 Tableau de bord des analyses")

    st.write("Voici l'historique d'utilisation :")
    st.dataframe(historique)

    st.write("Graphique du nombre d'analyses par type :")
    st.bar_chart(historique["type"].value_counts())
