import streamlit as st
from transformers import pipeline
import warnings

warnings.simplefilter(action='ignore')

# Carga del modelo de pregunta-respuesta en español
@st.cache_resource
def load_qa_model():
    return pipeline(
        'question-answering',
        model='mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es',
        tokenizer=('mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es', {'use_fast': True})
    )

qa_model = load_qa_model()

# Interfaz de usuario
st.title("🔍 Análisis de Tweets con Transformers")
st.markdown("Esta app responde preguntas sobre texto (como tweets) usando NLP y Hugging Face.")

# Entrada de texto
context = st.text_area("📝 Escribe uno o varios tweets como contexto:", height=200)
question = st.text_input("❓ ¿Qué querés preguntar sobre ese texto?")

# Botón para procesar
if st.button("Responder"):
    if context.strip() == "" or question.strip() == "":
        st.warning("Por favor, completá tanto el contexto como la pregunta.")
    else:
        result = qa_model(question=question, context=context)
        st.success(f"💡 Respuesta: {result['answer']}")
        st.caption(f"(Con una puntuación de confianza de {round(result['score'] * 100, 2)}%)")
