from PyPDF2 import PdfReader
import os
import spacy
import streamlit as st

# librerías para el procesamiento de lenguaje natural
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from streamlit_chat import message

# Configuración de Streamlit
st.set_page_config(page_title="Chatbot con PDF", layout="wide")
st.markdown("""<style>.block-container {padding-top: 1rem;}</style>""", unsafe_allow_html=True)

# Set OPENAI API KEY
OPENAI_API_KEY = "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"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Creando las llaves para el session_state
session_state = {
    "responses": [],
    "requests": []
}

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["Que onda Viejon, En que lo ayudo compa?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

# Función para extraer texto del PDF
def extract_text_from_pdf(pdf):
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

# Interfaz de usuario de Streamlit
st.sidebar.markdown("<h1 style ='text-align: center; color: #176887;'>Cargar Archivo PDF</h1>", unsafe_allow_html=True)
st.sidebar.write("Por favor, carga el archivo PDF con el cual quieres interactuar")
pdf_doc = st.sidebar.file_uploader("", type="pdf")
st.sidebar.write("---")
clear_button = st.sidebar.button("Limpiar Conversación", key="clear")

# Crear embeddings solo si pdf_doc está definido
if pdf_doc is not None:
    pdf_text = extract_text_from_pdf(pdf_doc)
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=10000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(pdf_text)
    embeddings = OpenAIEmbeddings()
    embeddings_pdf = FAISS.from_texts(chunks, embeddings)

    # Analizar el texto con spaCy para identificar entidades clave
    spacy.cli.download("es_core_news_sm")
    nlp = spacy.load("es_core_news_sm")
    doc = nlp(pdf_text)

    items_to_quote = []
    for ent in doc.ents:
        if ent.label_ == "PRODUCTO":  # Ajusta esto según tus necesidades
            items_to_quote.append(ent.text)

# CHAT SECTION
st.markdown("<h2 style='text-align: center; color: #176B87; text-decoration: underline;' ><strong>Interactua con Kelvinator 2000 sobre tu documento</strong></h2>", unsafe_allow_html=True)
st.write("---")
response_container = st.container()
textcontainer = st.container()

with textcontainer:
    with st.form(key='my_form', clear_on_submit=True):
        query = st.text_area("Tu:", key='input', height=100)
        submit_button = st.form_submit_button(label='Enviar')

    if query:
        with st.spinner("escribiendo..."):
            docs = embeddings_pdf.similarity_search(query)
            llm = OpenAI(model_name="gpt-3.5-turbo-1106")
            chain = load_qa_chain(llm, chain_type="stuff")

            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)

        st.session_state.requests.append(query)
        st.session_state.responses.append(response)

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i), avatar_style="pixel-art")
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i)+'_user')

# Mostrar elementos a cotizar en la interfaz
if pdf_doc is not None:
    st.write(f"Elementos a cotizar: {', '.join(items_to_quote)}")
