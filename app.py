import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import platform

# Estilos minimalistas
st.markdown("""
<style>
  /* Fondo blanco y texto siempre negro */
  body, html, .block-container, .stApp, .stSidebar, p, span, div, h1, h2, h3, h4, h5, h6, label {
    background-color: #454545 !important;
    color: #000000 !important;
  }
  /* Inputs y textareas */
  .stTextArea textarea, .stTextInput input {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 1px solid #ccc !important;
  }
  /* Botones */
  .stButton > button {
    background-color: #000000 !important;
    color: #ffffff !important;
    border-radius: 4px !important;
    padding: 0.5rem 1rem !important;
  }
  .stButton > button:hover {
    background-color: #333333 !important;
  }
</style>
""", unsafe_allow_html=True)

# Título y versión
st.title('Generación Aumentada por Recuperación (RAG) 💬')
st.write("Versión de Python:", platform.python_version())

# Cargar y mostrar logo
try:
    img = Image.open('Chat_pdf.png')
    st.image(img, width=350)
except:
    pass

# Sidebar
with st.sidebar:
    st.subheader("Ayuda RAG con tu PDF")
    st.write("Carga un PDF, luego formula preguntas o mira el resumen y keywords generados automáticamente.")

# Clave de API
api_key = st.text_input('Clave de OpenAI', type='password')
if api_key:
    os.environ['OPENAI_API_KEY'] = api_key
else:
    st.warning('Necesitas tu clave de OpenAI para continuar')

# Carga de PDF
pdf_file = st.file_uploader('Carga un archivo PDF', type='pdf')

if pdf_file is not None and api_key:
    try:
        # Extraer texto
        reader = PdfReader(pdf_file)
        full_text = ''.join(page.extract_text() for page in reader.pages if page.extract_text())
        st.info(f'Texto extraído: {len(full_text)} caracteres')

        # Generar resumen y keywords con LLM
        llm = OpenAI(temperature=0.3, model_name='gpt-4o')
        prompt = (
            'Por favor, proporciona un resumen breve del siguiente texto y lista las palabras clave principales.\n\n'
            f'{full_text[:2000]}'
        )
        st.markdown('### 📑 Resumen y Keywords')
        response = llm(prompt)
        st.markdown(response)

        # Dividir en fragmentos y construir base de conocimiento
        splitter = CharacterTextSplitter(separator='\n', chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(full_text)
        st.success(f'Documento dividido en {len(chunks)} fragmentos')
        embeddings = OpenAIEmbeddings()
        kb = FAISS.from_texts(chunks, embeddings)

        # Preguntas del usuario
        st.subheader('❓ Pregunta al documento')
        question = st.text_area('', placeholder='Escribe tu pregunta aquí...')
        if question:
            docs = kb.similarity_search(question)
            chain = load_qa_chain(llm, chain_type='stuff')
            answer = chain.run(input_documents=docs, question=question)
            st.markdown('### 📝 Respuesta:')
            st.markdown(answer)

    except Exception as e:
        st.error(f'Error procesando PDF: {e}')
else:
    if pdf_file and not api_key:
        st.warning('Ingresa tu clave de OpenAI.')
    else:
        st.info('Por favor carga un PDF para comenzar.')
