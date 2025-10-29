# Import Langchain dependencies
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, CSVLoader
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Other imports
import streamlit as st
import tempfile
import os

# Load API key from environment variable
import os
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("⚠️ GROQ API key not found. Please get one from https://console.groq.com and set the 'GROQ_API_KEY' environment variable.")
    st.stop()

# Setup LLM - Using Groq API (fast and reliable)
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=api_key,
    temperature=0.3,  # Más bajo para respuestas más precisas y menos creativas
    max_tokens=1024  # Más tokens para respuestas completas
)

# Streamlit UI
st.set_page_config(page_title="Ask RAG - USAT", page_icon="🤖", layout="wide")
st.title("🤖 Ask RAG - Multi-file Support")
st.caption("Powered by Groq & LangChain | Asistente inteligente para documentos USAT")

# Upload multiple files
uploaded_files = st.file_uploader(
    "📂 Upload files (PDF, DOCX, TXT, CSV)", 
    type=["pdf", "docx", "txt", "csv"], 
    accept_multiple_files=True,
    help="Sube uno o varios documentos para hacer preguntas sobre su contenido"
)

# Function to load and process multiple files
@st.cache_resource
def load_files(files):
    if not files:
        return None  # No files uploaded
    
    loaders = []
    temp_files = []

    # Process each file
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[-1]) as temp_file:
            temp_file.write(file.read())
            temp_path = temp_file.name
            temp_files.append(temp_path)  # Store paths to delete later
        
        # Detect file type and use appropriate loader
        if file.name.endswith(".pdf"):
            loaders.append(PyPDFLoader(temp_path))
        elif file.name.endswith(".txt"):
            loaders.append(TextLoader(temp_path))
        elif file.name.endswith(".docx"):
            loaders.append(UnstructuredWordDocumentLoader(temp_path))
        elif file.name.endswith(".csv"):
            loaders.append(CSVLoader(temp_path))

    # Load documents from all loaders with metadata
    documents = []
    for idx, loader in enumerate(loaders):
        docs = loader.load()
        # Add metadata to identify source document
        for doc in docs:
            doc.metadata['source_file'] = files[idx].name
            doc.metadata['file_type'] = os.path.splitext(files[idx].name)[-1]
            doc.metadata['file_index'] = idx
        documents.extend(docs)
    
    # Split documents into chunks with better parameters for context retention
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Mayor tamaño para mejor contexto
        chunk_overlap=200,  # Mayor overlap para continuidad
        separators=["\n\n", "\n", ". ", " ", ""]  # Separadores inteligentes
    )
    texts = text_splitter.split_documents(documents)
    
    # Log document processing info
    st.info(f"📄 Procesados {len(documents)} documentos en {len(texts)} fragmentos")
    
    # Create embeddings and vector store with search optimization
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}  # Mejora la precisión de búsqueda
    )
    vectorstore = FAISS.from_documents(texts, embeddings)

    return vectorstore, temp_files

# Load files only if any files are uploaded
if uploaded_files:
    with st.spinner("📚 Procesando documentos..."):
        vectorstore, temp_files = load_files(uploaded_files)
    st.success(f"✅ {len(uploaded_files)} archivo(s) cargado(s) correctamente!")
else:
    vectorstore, temp_files = None, []

# Initialize Q&A chain if the vectorstore is created
if vectorstore:
    from langchain_core.prompts import PromptTemplate
    custom_prompt = PromptTemplate(
        template="""
        Eres Usatín, un asistente virtual amable, carismático y servicial de la Universidad USAT.
        
        REGLAS DE PRESENTACIÓN:
        - Si detectas "[PRIMERA INTERACCIÓN]" en la pregunta, preséntate diciendo: "¡Hola! 😊 Soy Usatín, tu asistente virtual de la USAT. Estoy aquí para ayudarte con cualquier duda sobre nuestros documentos. ¿En qué puedo ayudarte hoy?"
        - Luego responde la pregunta (eliminando el marcador [PRIMERA INTERACCIÓN] de tu análisis)
        - En mensajes posteriores, NO te presentes de nuevo, solo responde de forma amable y directa
        
        PERSONALIDAD:
        - Sé cálido, amable y entusiasta
        - Usa emojis ocasionales para ser más expresivo (😊, 📚, ✅, 💡, etc.)
        - Habla de forma natural y cercana
        - Muestra interés genuino por ayudar
        - Sé profesional pero no rígido
        
        INSTRUCCIONES DE RESPUESTA:
        1. Responde SOLO usando la información del contexto proporcionado
        2. Si no encuentras la respuesta, di amablemente: "Lo siento, no encontré esa información en los documentos que tengo disponibles 😔. ¿Hay algo más en lo que pueda ayudarte?"
        3. Si encuentras información parcial, indícalo con empatía
        4. Mantén las respuestas claras, bien estructuradas y fáciles de entender
        5. Si hay tablas o listas, respétalas y preséntalalas de forma ordenada
        6. Termina tus respuestas invitando a hacer más preguntas cuando sea apropiado

        Contexto de los documentos:
        {context}

        Pregunta del usuario:
        {question}
        
        Respuesta (recuerda ser amable y carismático):
        """,
        input_variables=["context", "question"]
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}  # Recupera 4 documentos más relevantes
        ),
        input_key="question",
        return_source_documents=True,  # Ahora retornamos fuentes
        chain_type_kwargs={"prompt": custom_prompt}
    )

    # Setup session state messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Setup first interaction flag
    if 'first_interaction' not in st.session_state:
        st.session_state.first_interaction = True

    # Display previous chat messages
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])

    # User input
    prompt = st.chat_input("Escribe tu pregunta aquí...")

    if prompt:
        # Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Add greeting to first message
        question_to_process = prompt
        if st.session_state.first_interaction:
            question_to_process = f"[PRIMERA INTERACCIÓN] {prompt}"
            st.session_state.first_interaction = False

        # Process prompt with LLM - Using invoke instead of run
        with st.spinner("🤔 Pensando..."):
            response = chain.invoke({"question": question_to_process})
            answer = response.get("result", "No answer generated")
            source_docs = response.get("source_documents", [])

        # Display assistant response
        st.chat_message("assistant").markdown(answer)
        
        # Show source documents used
        if source_docs:
            with st.expander("📚 Fuentes consultadas"):
                sources_seen = set()
                for doc in source_docs:
                    source_file = doc.metadata.get('source_file', 'Desconocido')
                    if source_file not in sources_seen:
                        st.markdown(f"- **{source_file}**")
                        sources_seen.add(source_file)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
        
else:
    st.info("👆 Por favor, sube archivos para comenzar a hacer preguntas.")
    st.markdown("""
    ### 📖 Cómo usar:
    1. Sube uno o más documentos (PDF, DOCX, TXT, CSV)
    2. Espera a que se procesen
    3. Haz preguntas sobre el contenido
    4. Recibe respuestas basadas únicamente en los documentos
    """)

# Button to clear all messages and reset file upload
if st.button("🗑️ Limpiar todo", type="secondary"):
    st.session_state.messages = []
    st.session_state.first_interaction = True  # Reset first interaction
    uploaded_files = None  # Reset file uploader

    # Delete temporary files
    for file_path in temp_files:
        try:
            os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")

    # Reload the app
    st.rerun()
