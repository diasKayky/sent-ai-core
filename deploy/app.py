import streamlit as st
import tensorflow.keras.backend as K
from ensemble_model import SentAI

st.title("Sent.IA")
st.markdown("Sent.AI é um modelo de Machine Learning que utiliza rede neural LSTM (memória de curto longo prazo) para analisar sentimento de textos e classificar em 'positivo' ou 'negativo'.")

st.subheader("Conjunto de Dados para Analisar:")
arquivo = st.file_uploader(label="Faça upload:")

if st.button("Enviar para predição"):
    st.write("Nice try!")
