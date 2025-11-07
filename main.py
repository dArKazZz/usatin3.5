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
import hashlib
import pickle
from datetime import datetime
from pathlib import Path
import base64
from io import BytesIO
import re

# Voice imports
try:
    import speech_recognition as sr
    from gtts import gTTS
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False
    st.warning("⚠️ Librerías de voz no instaladas. Instala: pip install SpeechRecognition gtts pyaudio")

# Load API key from environment variable
import os
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("⚠️ GROQ API key not found. Please get one from https://console.groq.com and set the 'GROQ_API_KEY' environment variable.")
    st.stop()

# Setup LLM - Using Groq API (fast and reliable)
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=api_key,  # Usa la variable de entorno
    temperature=0.3,  # Más bajo para respuestas más precisas y menos creativas
    max_tokens=2048  # Más tokens para respuestas completas con streaming
)

# Streamlit UI - Enhanced Design
st.set_page_config(
    page_title="Usatín - Asistente Inteligente USAT",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern, clean interface
st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }
    
    /* Content area */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        background: white;
        border-radius: 20px;
        margin: 2rem auto;
        max-width: 1200px;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    
    /* Title styling */
    h1 {
        color: #667eea;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        font-size: 2.5rem !important;
    }
    
    /* Caption styling */
    .stCaption {
        text-align: center;
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    
    /* File uploader styling */
    .stFileUploader {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-radius: 15px;
        padding: 1.5rem;
        border: 2px dashed #667eea;
        margin-bottom: 1.5rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Chat input */
    .stChatInputContainer {
        border-radius: 15px;
        border: 2px solid #667eea30;
        background: white;
    }
    
    /* Chat messages */
    .stChatMessage {
        background: #f8f9ff;
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-radius: 10px;
        font-weight: 600;
        color: #667eea;
    }
    
    /* Success message */
    .stSuccess {
        background: linear-gradient(135deg, #11998e15 0%, #38ef7d15 100%);
        border-left: 4px solid #11998e;
        border-radius: 10px;
    }
    
    /* Error message */
    .stError {
        background: linear-gradient(135deg, #eb3349 15 0%, #f45c43 15 100%);
        border-left: 4px solid #eb3349;
        border-radius: 10px;
    }
    
    /* Checkbox */
    .stCheckbox {
        background: #f8f9ff;
        padding: 0.5rem;
        border-radius: 8px;
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(17, 153, 142, 0.4);
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Divider */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header with icon and title
st.markdown("""
    <div style='text-align: center; padding: 1rem 0;'>
        <h1>🤖 Usatín</h1>
        <p style='font-size: 1.2rem; color: #666; margin: 0;'>
            Tu Asistente Inteligente USAT
        </p>
        <p style='font-size: 0.9rem; color: #999; margin-top: 0.5rem;'>
            💡 Powered by Groq & LangChain | Con reconocimiento de voz en español
        </p>
    </div>
""", unsafe_allow_html=True)

st.markdown("---")

# Upload section with better styling
st.markdown("### 📂 Sube tus Documentos")
uploaded_files = st.file_uploader(
    "Arrastra archivos aquí o haz clic para seleccionar", 
    type=["pdf", "docx", "txt", "csv"], 
    accept_multiple_files=True,
    help="Formatos soportados: PDF, Word, Texto, CSV",
    label_visibility="collapsed"
)

# Function to create hash of file contents for caching
def get_files_hash(files):
    """Generate hash based on file contents for proper caching"""
    if not files:
        return None
    hash_content = ""
    for file in files:
        file.seek(0)  # Reset file pointer
        hash_content += f"{file.name}_{hashlib.md5(file.read()).hexdigest()}"
        file.seek(0)  # Reset again for reading
    return hashlib.md5(hash_content.encode()).hexdigest()

# Function to load and process multiple files
def load_files(files, files_hash):
    if not files:
        return None, []  # No files uploaded
    
    # Check if vectorstore exists in cache directory
    cache_dir = Path("./vectorstore_cache")
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / f"{files_hash}.pkl"
    
    # Try to load from cache
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)
                st.info(f"✨ Cargado desde caché - {len(files)} archivo(s)")
                return cached_data['vectorstore'], cached_data['temp_files']
        except Exception as e:
            st.warning(f"⚠️ Error al cargar caché, reprocesando... ({str(e)})")
    
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

    # Load documents from all loaders with enriched metadata
    documents = []
    for idx, loader in enumerate(loaders):
        try:
            docs = loader.load()
            # Add enriched metadata to identify source document
            for doc_idx, doc in enumerate(docs):
                doc.metadata['source_file'] = files[idx].name
                doc.metadata['file_type'] = os.path.splitext(files[idx].name)[-1]
                doc.metadata['file_index'] = idx
                doc.metadata['doc_index'] = doc_idx
                doc.metadata['upload_time'] = datetime.now().isoformat()
                # Add page number if available (for PDFs)
                if 'page' in doc.metadata:
                    doc.metadata['page_number'] = doc.metadata['page'] + 1
            documents.extend(docs)
        except Exception as e:
            st.error(f"❌ Error procesando {files[idx].name}: {str(e)}")
            continue
    
    if not documents:
        st.error("❌ No se pudieron cargar documentos")
        return None, temp_files
    
    # Split documents into chunks with optimized parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # Mayor tamaño para mejor contexto
        chunk_overlap=300,  # Mayor overlap para continuidad
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],  # Separadores inteligentes
        length_function=len,
    )
    texts = text_splitter.split_documents(documents)
    
    # Add chunk metadata
    for idx, text in enumerate(texts):
        text.metadata['chunk_index'] = idx
        text.metadata['chunk_length'] = len(text.page_content)
    
    # Log document processing info
    st.info(f"📄 Procesados {len(documents)} documentos en {len(texts)} fragmentos optimizados")
    
    # Create embeddings and vector store with search optimization
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}  # Mejora la precisión de búsqueda
        )
        vectorstore = FAISS.from_documents(texts, embeddings)
        
        # Save to cache
        try:
            cache_data = {
                'vectorstore': vectorstore,
                'temp_files': temp_files,
                'created_at': datetime.now().isoformat(),
                'num_docs': len(documents),
                'num_chunks': len(texts)
            }
            with open(cache_path, "wb") as f:
                pickle.dump(cache_data, f)
            st.success(f"💾 Vectorstore guardado en caché")
        except Exception as e:
            st.warning(f"⚠️ No se pudo guardar en caché: {str(e)}")
        
    except Exception as e:
        st.error(f"❌ Error creando embeddings: {str(e)}")
        return None, temp_files

    return vectorstore, temp_files

# Voice functions
def record_audio():
    """Record audio from microphone and convert to text using Google Speech Recognition"""
    if not VOICE_AVAILABLE:
        st.error("❌ Funcionalidad de voz no disponible. Instala las dependencias necesarias.")
        return None
    
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("🎤 Escuchando... Habla ahora")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            st.success("✅ Audio grabado, procesando...")
            
            # Recognize speech using Google Speech Recognition (Spanish)
            text = recognizer.recognize_google(audio, language="es-ES")
            return text
    except sr.WaitTimeoutError:
        st.error("❌ Tiempo de espera agotado. No se detectó voz.")
        return None
    except sr.UnknownValueError:
        st.error("❌ No se pudo entender el audio. Intenta hablar más claro.")
        return None
    except sr.RequestError as e:
        st.error(f"❌ Error con el servicio de reconocimiento: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Error al grabar audio: {e}")
        return None

def text_to_speech(text):
    """Convert text to speech in Spanish and return audio"""
    if not VOICE_AVAILABLE:
        return None
    
    try:
        # Clean markdown formatting from text
        clean_text = text
        
        # Remove bold/italic markers (**, *, __, _)
        clean_text = re.sub(r'\*\*\*(.+?)\*\*\*', r'\1', clean_text)  # ***text***
        clean_text = re.sub(r'\*\*(.+?)\*\*', r'\1', clean_text)      # **text**
        clean_text = re.sub(r'\*(.+?)\*', r'\1', clean_text)          # *text*
        clean_text = re.sub(r'___(.+?)___', r'\1', clean_text)        # ___text___
        clean_text = re.sub(r'__(.+?)__', r'\1', clean_text)          # __text__
        clean_text = re.sub(r'_(.+?)_', r'\1', clean_text)            # _text_
        
        # Remove standalone asterisks and underscores
        clean_text = clean_text.replace('*', '').replace('_', '')
        
        # Remove markdown headers (# ## ###)
        clean_text = re.sub(r'^#{1,6}\s+', '', clean_text, flags=re.MULTILINE)
        
        # Remove markdown links [text](url)
        clean_text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', clean_text)
        
        # Remove markdown images ![alt](url)
        clean_text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', r'\1', clean_text)
        
        # Remove inline code `code`
        clean_text = re.sub(r'`([^`]+)`', r'\1', clean_text)
        
        # Remove code blocks ```code```
        clean_text = re.sub(r'```[^\n]*\n(.*?)```', r'\1', clean_text, flags=re.DOTALL)
        
        # Remove horizontal rules (---, ___, ***)
        clean_text = re.sub(r'^[\-_\*]{3,}$', '', clean_text, flags=re.MULTILINE)
        
        # Remove bullet points (-, *, +)
        clean_text = re.sub(r'^\s*[\-\*\+]\s+', '', clean_text, flags=re.MULTILINE)
        
        # Remove numbered lists (1. 2. etc)
        clean_text = re.sub(r'^\s*\d+\.\s+', '', clean_text, flags=re.MULTILINE)
        
        # Remove blockquotes (>)
        clean_text = re.sub(r'^\s*>\s+', '', clean_text, flags=re.MULTILINE)
        
        # Remove emojis
        # Emoji pattern covers most common emoji ranges
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"  # dingbats
            "\U000024C2-\U0001F251"  # enclosed characters
            "\U0001F900-\U0001F9FF"  # supplemental symbols
            "\U0001FA00-\U0001FA6F"  # chess symbols
            "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-a
            "\U00002600-\U000026FF"  # miscellaneous symbols
            "\U00002B50"              # star
            "\U0001F004"              # mahjong tile
            "]+", 
            flags=re.UNICODE
        )
        clean_text = emoji_pattern.sub('', clean_text)
        
        # Remove extra whitespace
        clean_text = re.sub(r'\n\s*\n', '\n\n', clean_text)
        clean_text = clean_text.strip()
        
        # Create gTTS object with cleaned Spanish text
        tts = gTTS(text=clean_text, lang='es', slow=False)
        
        # Save to BytesIO object
        audio_bytes = BytesIO()
        tts.write_to_fp(audio_bytes)
        audio_bytes.seek(0)
        
        return audio_bytes.read()
    except Exception as e:
        st.error(f"❌ Error generando audio: {e}")
        return None

def autoplay_audio(audio_bytes):
    """Autoplay audio in Streamlit"""
    if audio_bytes:
        b64 = base64.b64encode(audio_bytes).decode()
        audio_html = f"""
            <audio autoplay>
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)

# Load files only if any files are uploaded
if uploaded_files:
    files_hash = get_files_hash(uploaded_files)
    with st.spinner("📚 Procesando documentos..."):
        vectorstore, temp_files = load_files(uploaded_files, files_hash)
    if vectorstore:
        st.success(f"✅ {len(uploaded_files)} archivo(s) cargado(s) correctamente!")
    else:
        st.error("❌ Error al cargar los archivos")
        vectorstore, temp_files = None, []
else:
    vectorstore, temp_files = None, []

# Initialize Q&A chain if the vectorstore is created
if vectorstore:
    from langchain_core.prompts import PromptTemplate
    
    # Function for query expansion
    def expand_query(question):
        """Expand query with synonyms and reformulations for better retrieval"""
        expansions = []
        
        # Original question
        expansions.append(question)
        
        # Add variations for common academic terms
        if "qué es" in question.lower() or "que es" in question.lower():
            expansions.append(question.replace("qué es", "definición de").replace("que es", "definición de"))
            expansions.append(question.replace("qué es", "concepto de").replace("que es", "concepto de"))
        
        if "cómo" in question.lower() or "como" in question.lower():
            expansions.append(question.replace("cómo", "proceso para").replace("como", "proceso para"))
            expansions.append(question.replace("cómo", "pasos para").replace("como", "pasos para"))
        
        return expansions
    
    # Custom retriever with re-ranking
    def retrieve_with_reranking(question, k=10):
        """Retrieve documents with query expansion and re-ranking"""
        
        # Expand query
        expanded_queries = expand_query(question)
        
        # Retrieve more documents initially
        all_docs = []
        seen_content = set()
        
        for query in expanded_queries[:2]:  # Use top 2 variations
            docs = vectorstore.similarity_search(query, k=k)
            for doc in docs:
                # Avoid duplicates
                content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    all_docs.append(doc)
        
        # Simple scoring based on keyword matches and position
        scored_docs = []
        question_words = set(question.lower().split())
        
        for doc in all_docs:
            doc_words = set(doc.page_content.lower().split())
            # Calculate overlap score
            overlap = len(question_words & doc_words)
            # Boost score for documents with metadata
            metadata_boost = 0
            if 'page_number' in doc.metadata:
                metadata_boost += 0.5
            if doc.metadata.get('chunk_index', float('inf')) < 5:
                metadata_boost += 0.3  # Boost early chunks
            
            score = overlap + metadata_boost
            scored_docs.append((score, doc))
        
        # Sort by score and return top k
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        return [doc for score, doc in scored_docs[:k]]
    
    custom_prompt = PromptTemplate(
        template="""
        Eres Usatín, un asistente virtual amable, carismático y servicial de la Universidad USAT.
        
        
        PERSONALIDAD:
        - Sé cálido, amable y entusiasta
        - Usa emojis ocasionales para ser más expresivo
        - Habla de forma natural y cercana
        - Muestra interés genuino por ayudar
        - Sé profesional pero no rígido
        
        INSTRUCCIONES DE RESPUESTA:
        1. Responde SOLO usando la información del contexto proporcionado
        2. Si no encuentras la respuesta, di amablemente: "Lo siento, esa información no está disponible. ¿Hay algo más en lo que pueda ayudarte?"
        3. Si encuentras información parcial, indícalo con empatía
        4. Mantén las respuestas claras, bien estructuradas y fáciles de entender
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
            search_kwargs={"k": 10}  # Aumentado para mejor cobertura
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
    
    # Generate suggested questions based on documents
    if 'suggested_questions' not in st.session_state and vectorstore:
        with st.spinner("💡 Generando preguntas sugeridas..."):
            try:
                # Get sample documents
                sample_docs = vectorstore.similarity_search("información principal contenido documento", k=3)
                context_sample = "\n\n".join([doc.page_content[:500] for doc in sample_docs])
                
                # Generate questions using LLM
                suggestion_prompt = f"""Basándote en este contenido de documentos, genera 3 preguntas específicas y útiles que un usuario podría hacer.
Devuelve SOLO las preguntas, una por línea, sin numeración ni explicaciones adicionales.

Contenido:
{context_sample[:1000]}

Preguntas:"""
                
                suggestions_response = llm.invoke(suggestion_prompt)
                questions = suggestions_response.content.strip().split('\n')
                questions = [q.strip().lstrip('0123456789.-) ') for q in questions if q.strip()][:3]
                st.session_state.suggested_questions = questions
            except Exception as e:
                st.session_state.suggested_questions = []
    
    # Display suggested questions with enhanced styling
    if hasattr(st.session_state, 'suggested_questions') and st.session_state.suggested_questions:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); 
                        padding: 1.5rem; border-radius: 15px; margin: 1.5rem 0;'>
                <h3 style='color: #667eea; margin-bottom: 1rem; font-size: 1.3rem;'>
                    💡 Preguntas Sugeridas
                </h3>
            </div>
        """, unsafe_allow_html=True)
        
        cols = st.columns(len(st.session_state.suggested_questions))
        for idx, (col, question) in enumerate(zip(cols, st.session_state.suggested_questions)):
            with col:
                if st.button(question[:60] + ("..." if len(question) > 60 else ""), 
                           key=f"suggestion_{idx}", 
                           use_container_width=True,
                           help=question):
                    st.session_state.clicked_question = question
    
    st.markdown("---")

    # Display previous chat messages
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])
    
    # Voice input section with modern design
    if VOICE_AVAILABLE:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #11998e15 0%, #38ef7d15 100%); 
                        padding: 1.5rem; border-radius: 15px; margin: 1.5rem 0; text-align: center;'>
                <h3 style='color: #11998e; margin-bottom: 0.5rem; font-size: 1.2rem;'>
                    🎙️ Interacción por Voz
                </h3>
                <p style='color: #666; margin: 0; font-size: 0.9rem;'>
                    Habla con Usatín usando tu micrófono
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        col_voice1, col_voice2, col_voice3 = st.columns([1, 2, 1])
        with col_voice2:
            voice_col1, voice_col2 = st.columns(2)
            with voice_col1:
                if st.button("🎤 Grabar Pregunta", use_container_width=True, type="primary"):
                    transcribed_text = record_audio()
                    if transcribed_text:
                        st.session_state.voice_input = transcribed_text
                        st.success(f"✅ '{transcribed_text}'")
                        st.rerun()
            
            with voice_col2:
                if 'auto_voice' not in st.session_state:
                    st.session_state.auto_voice = False
                auto_voice = st.checkbox("🔊 Respuestas con voz", 
                                       value=st.session_state.auto_voice,
                                       help="Activa para escuchar las respuestas")
                st.session_state.auto_voice = auto_voice
    
    # Export conversation functionality with better positioning
    if st.session_state.messages:
        col_exp1, col_exp2, col_exp3 = st.columns([3, 2, 3])
        with col_exp2:
            # Prepare conversation text
            conversation_text = f"Conversación con Usatín - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            conversation_text += "=" * 60 + "\n\n"
            for msg in st.session_state.messages:
                role = "Usuario" if msg["role"] == "user" else "Usatín"
                conversation_text += f"{role}:\n{msg['content']}\n\n"
                conversation_text += "-" * 60 + "\n\n"
            
            st.download_button(
                label="📥 Exportar chat",
                data=conversation_text,
                file_name=f"conversacion_usatin_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                help="Descarga la conversación completa en formato texto"
            )
    
    st.markdown("---")
    
    # User input
    prompt = st.chat_input("Escribe tu pregunta aquí...")
    
    # Handle voice input
    if 'voice_input' in st.session_state:
        prompt = st.session_state.voice_input
        del st.session_state.voice_input
    
    # Handle clicked suggestion
    if 'clicked_question' in st.session_state:
        prompt = st.session_state.clicked_question
        del st.session_state.clicked_question

    if prompt:
        # Display user message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Add greeting to first message
        question_to_process = prompt
        if st.session_state.first_interaction:
            question_to_process = f"[PRIMERA INTERACCIÓN] {prompt}"
            st.session_state.first_interaction = False

        # Process prompt with LLM - Using invoke with streaming
        with st.spinner("🤔 Pensando..."):
            try:
                # Get documents with re-ranking
                retrieved_docs = retrieve_with_reranking(question_to_process, k=8)
                
                # Process with chain
                response = chain.invoke({
                    "question": question_to_process,
                })
                answer = response.get("result", "No answer generated")
                source_docs = response.get("source_documents", [])
            except Exception as e:
                st.error(f"❌ Error al procesar la pregunta: {str(e)}")
                answer = "Lo siento, hubo un error al procesar tu pregunta. Por favor, intenta de nuevo."
                source_docs = []

        # Display assistant response with streaming effect
        message_placeholder = st.chat_message("assistant")
        full_response = ""
        
        # Simulate streaming by displaying word by word
        words = answer.split()
        displayed_text = message_placeholder.empty()
        for i, word in enumerate(words):
            full_response += word + " "
            displayed_text.markdown(full_response + "▌")
            # Small delay for streaming effect (adjust as needed)
            import time
            time.sleep(0.03)
        displayed_text.markdown(full_response)
        
        # Show detailed source documents used
        if source_docs:
            with st.expander("📚 Fuentes consultadas (con fragmentos)"):
                for idx, doc in enumerate(source_docs[:5], 1):  # Limit to top 5 sources
                    source_file = doc.metadata.get('source_file', 'Desconocido')
                    page_num = doc.metadata.get('page_number', 'N/A')
                    chunk_idx = doc.metadata.get('chunk_index', 'N/A')
                    
                    st.markdown(f"**{idx}. {source_file}** (Página: {page_num}, Fragmento: {chunk_idx})")
                    # Show preview of content used
                    preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                    st.markdown(f"> {preview}")
                    st.markdown("---")
        
        # Generate and play voice response if enabled
        if VOICE_AVAILABLE and st.session_state.get('auto_voice', False):
            with st.spinner("🔊 Generando audio..."):
                audio_bytes = text_to_speech(full_response)
                if audio_bytes:
                    autoplay_audio(audio_bytes)
                    # Also provide manual download option
                    st.audio(audio_bytes, format='audio/mp3')
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
else:
    # Welcome screen with modern design
    st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); 
                    padding: 2rem; border-radius: 20px; margin: 2rem 0; text-align: center;'>
            <h2 style='color: #667eea; margin-bottom: 1rem;'>
                👋 ¡Bienvenido a Usatín!
            </h2>
            <p style='font-size: 1.1rem; color: #666; margin-bottom: 2rem;'>
                Tu asistente inteligente para documentos institucionales
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div style='background: white; padding: 1.5rem; border-radius: 15px; 
                        border: 2px solid #667eea30; text-align: center; height: 100%;'>
                <div style='font-size: 3rem; margin-bottom: 1rem;'>📄</div>
                <h3 style='color: #667eea; font-size: 1.2rem; margin-bottom: 0.5rem;'>
                    Multi-formato
                </h3>
                <p style='color: #666; font-size: 0.9rem;'>
                    Soporta PDF, Word, TXT y CSV
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style='background: white; padding: 1.5rem; border-radius: 15px; 
                        border: 2px solid #11998e30; text-align: center; height: 100%;'>
                <div style='font-size: 3rem; margin-bottom: 1rem;'>🎙️</div>
                <h3 style='color: #11998e; font-size: 1.2rem; margin-bottom: 0.5rem;'>
                    Interacción por Voz
                </h3>
                <p style='color: #666; font-size: 0.9rem;'>
                    Habla y escucha respuestas
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div style='background: white; padding: 1.5rem; border-radius: 15px; 
                        border: 2px solid #f093fb30; text-align: center; height: 100%;'>
                <div style='font-size: 3rem; margin-bottom: 1rem;'>🎯</div>
                <h3 style='color: #f093fb; font-size: 1.2rem; margin-bottom: 0.5rem;'>
                    Respuestas Precisas
                </h3>
                <p style='color: #666; font-size: 0.9rem;'>
                    Basadas en tus documentos
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Instructions
    st.markdown("""
        <div style='background: #f8f9ff; padding: 2rem; border-radius: 15px; 
                    border-left: 4px solid #667eea;'>
            <h3 style='color: #667eea; margin-bottom: 1rem;'>
                🚀 Comienza en 3 pasos
            </h3>
            <ol style='color: #666; font-size: 1rem; line-height: 2;'>
                <li><strong>Sube tus documentos</strong> usando el selector de archivos arriba</li>
                <li><strong>Espera el procesamiento</strong> (solo toma unos segundos)</li>
                <li><strong>Haz preguntas</strong> escribiendo o usando el botón de voz 🎤</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tips
    with st.expander("💡 Consejos para mejores resultados"):
        st.markdown("""
        - ✅ Usa preguntas específicas y claras
        - ✅ Los documentos con texto seleccionable funcionan mejor
        - ✅ Puedes subir múltiples archivos relacionados
        - ✅ El sistema recuerda el contexto de la conversación
        - ✅ Activa la voz para una experiencia manos libres
        """)

st.markdown("---")

# Button to clear all messages and reset file upload with better styling
st.markdown("""
    <div style='margin-top: 2rem;'>
    </div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 2, 2])
with col1:
    if st.button("🗑️ Limpiar conversación", use_container_width=True):
        st.session_state.messages = []
        st.session_state.first_interaction = True
        if 'suggested_questions' in st.session_state:
            del st.session_state.suggested_questions
        st.rerun()

with col2:
    if st.button("💾 Limpiar caché", use_container_width=True, help="Elimina archivos cacheados"):
        cache_dir = Path("./vectorstore_cache")
        if cache_dir.exists():
            import shutil
            try:
                shutil.rmtree(cache_dir)
                st.success("✅ Caché limpiado")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        st.rerun()

with col3:
    if st.button("ℹ️ Ayuda", use_container_width=True):
        st.info("""
        **Usatín** es tu asistente RAG (Retrieval-Augmented Generation).
        
        - Sube documentos en la parte superior
        - Haz preguntas por texto o voz
        - Recibe respuestas con fuentes verificables
        """)

# Footer
st.markdown("""
    <div style='text-align: center; padding: 2rem 0; color: #999; font-size: 0.8rem;'>
        <p>Desarrollado con ❤️ para USAT | Versión 3.5 con IA</p>
    </div>
""", unsafe_allow_html=True)

# Delete temporary files on cleanup
if temp_files:
    for file_path in temp_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            pass  # Silent cleanup
