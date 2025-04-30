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

# ----------------- Estilos minimalistas y paleta -----------------
st.markdown("""
<style>
  /* Fondo limpio y tipografía moderna */
  body, html, .block-container { background-color: #fefefe !important; color: #000000 !important; }
  /* Encabezados */
  h1 { color: #2c3e50; }
  h2, h3, h4 { color: #34495e; }
  /* Inputs y textareas */
  .stTextArea textarea, .stTextInput input { background-color: #ffffff !important; color: #000000 !important; border: 1px solid #ced4da !important; border-radius: 4px !important; padding: 0.5rem !important; }
  /* Botones destacados */
  .stButton > button {
    background-color: #3498db !important;
    color: #ffffff !important;
    border-radius: 6px !important;
    padding: 0.6rem 1.2rem !important;
    font-size: 1rem !important;
  }
  .stButton > button:hover {
    background-color: #2980b9 !important;
  }
  /* Sidebar */
  .stSidebar { background-color: #ecf0f1 !important; }
  /* Tags de keywords */
  .keyword-tag { display: inline-block; background-color: #ffeaa7; color: #2d3436; padding: 0.25rem 0.5rem; margin: 0.25rem; border-radius: 4px; font-size: 0.9rem; }
  /* Alertas texto en negro */
  div[data-testid="stError"] *, div[data-testid="stWarning"] *, div[data-testid="stInfo"] *, div[data-testid="stSuccess"] * {
    color: #000000 !important;
  }
</style>
""", unsafe_allow_html=True)

# ----------------- Encabezado -----------------
st.title('💬 Generación Aumentada por Recuperación (RAG)')
st.write(f"🖥️ Python: {platform.python_version()}")

# ----------------- Logo -----------------
try:
    img = Image.open('Chat_pdf.png')
    st.image(img, width=200)
except:
    pass

# ----------------- Sidebar -----------------
with st.sidebar:
    st.header("📚 Carga y Analiza PDF")
    st.write("1. Ingresa tu API key de OpenAI.\n2. Carga tu PDF.\n3. Obtén resumen, keywords y haz preguntas.")

# ----------------- API Key -----------------
api_key = st.text_input('🔑 Clave de OpenAI', type='password')
if api_key:
    os.environ['OPENAI_API_KEY'] = api_key
else:
    st.warning('🔒 Ingresa tu clave para continuar')

# ----------------- Carga de PDF -----------------
pdf_file = st.file_uploader('📄 Selecciona un archivo PDF', type='pdf')

if pdf_file and api_key:
    try:
        # Extracción de texto
        reader = PdfReader(pdf_file)
        full_text = ''.join(p.extract_text() or '' for p in reader.pages)
        char_count = len(full_text)
        page_count = len(reader.pages)
        st.info(f"📝 Texto: {char_count} caracteres | 📄 Páginas: {page_count}")

        # Resumen y Keywords con LLM
        llm = OpenAI(temperature=0.3, model_name='gpt-4o')
        prompt = (
            "Resume brevemente el siguiente texto y lista sus principales palabras clave.\n\n"
            f"{full_text[:2000]}"
        )
        st.subheader('📑 Resumen y Keywords')
        llm_response = llm(prompt)
        st.markdown(llm_response)
        # Extraer keywords si existen
        keywords = []
        if 'Palabras clave:' in llm_response:
            section = llm_response.split('Palabras clave:')[-1]
            keywords = [k.strip() for k in section.replace('\n', ',').split(',') if k.strip()]
        if keywords:
            st.markdown('**Keywords:**')
            for kw in keywords:
                st.markdown(f"<span class='keyword-tag'>{kw}</span>", unsafe_allow_html=True)

        # Construcción de la base de conocimiento
        splitter = CharacterTextSplitter(separator='\n', chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(full_text)
        st.success(f"✅ {len(chunks)} fragmentos generados")
        embeddings = OpenAIEmbeddings()
        kb = FAISS.from_texts(chunks, embeddings)

        # Interfaz de pregunta/respuesta
        st.subheader('❓ Pregunta al documento')
        question = st.text_area('', placeholder='Escribe tu pregunta y presiona Enter...')
        if question:
            docs = kb.similarity_search(question)
            chain = load_qa_chain(llm, chain_type='stuff')
            answer = chain.run(input_documents=docs, question=question)
            st.markdown('### 📝 Respuesta')
            st.info(answer)
    except Exception as e:
        st.error(f"⚠️ Error procesando PDF: {e}")
else:
    if pdf_file and not api_key:
        st.warning('🔑 Por favor ingresa tu clave de OpenAI.')
    else:
        st.info('⬆️ Carga un PDF para comenzar.')
