from PyPDF2 import PdfReader
import os


#liberias:
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

#Librerias para el textcontainer
from langchain.llms import OpenAI 
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

#Librerias para el response_container
from streamlit_chat import message

#Stream Lit
import streamlit as st


#Configuracion streamlit

st.set_page_config(page_title="Chatbot con PDF", layout="wide")
st.markdown("""<style>.block-container {padding-top: 1rem;}</style>""", unsafe_allow_html=True)

#Set OPENAI API KEY
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

##creando las llaves para la session_state

session_state ={
    "responses":[],
    "requests":[]
    }

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["Que onda Viejon, En que lo ayudo compa?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []



def create_embeddings(pdf):
    # Extrayendo texto del pdf
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Dividiendo en trozos el texto extraido del PDF
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=10000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        embeddings = OpenAIEmbeddings()

        embeddings_pdf = FAISS.from_texts(chunks, embeddings)

        return embeddings_pdf


#cargar el documento en el sidebar
st.sidebar.markdown("<h1 style ='text-align: center; color: #176887;'>Cargar Archivo PDF</h1>", unsafe_allow_html = True)
st.sidebar.write("Por favor, carga el archivo PDF con el cual quieres interactuar")
pdf_doc = st.sidebar.file_uploader("", type="pdf")
st.sidebar.write("---")
clear_button = st.sidebar.button("Limpiar Conversación", key="clear")
    
#create embeddings
embeddings_pdf = create_embeddings(pdf_doc)


#CHAT SECTION 

st.markdown("<h2 style='text-align: center; color: #176B87; text-decoration: underline;' ><strong>Interactua con Kelvinator 2000 sobre tu docuento</strong></h2>", unsafe_allow_html=True)
st.write("---")
# container del chat history
response_container = st.container()
# container del text box
textcontainer = st.container()

## Creando el campo para el ingreso de la pregunta del usuario
with textcontainer:
    #Formulario del text input
    with st.form(key='my_form', clear_on_submit=True):
        query = st.text_area("Tu:", key='input', height=100)
        submit_button = st.form_submit_button(label='Enviar')

    if query:
        with st.spinner("escribiendo..."):

            #cosine similarity with API Word Embeddings
            docs = embeddings_pdf.similarity_search(query)

            #respuesta: 4 posibles respuestas

            llm = OpenAI(model_name="text-davinci-003")
            chain = load_qa_chain(llm, chain_type="stuff")


            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)

        st.session_state.requests.append(query)
        st.session_state.responses.append(response)


##Configurando el campo response_container para pintar el historial del chat
with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i), avatar_style="pixel-art")
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i)+'_user')
