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

# Load API key from environment variable
import os
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("⚠️ GROQ API key not found. Please get one from https://console.groq.com and set the 'GROQ_API_KEY' environment variable.")
    st.stop()

# Setup LLM - Using Groq API (fast and reliable)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=api_key,  # Usa la variable de entorno
    temperature=0.3,  # Más bajo para respuestas más precisas y menos creativas
    max_tokens=2048  # Más tokens para respuestas completas con streaming
)

# Streamlit UI
st.set_page_config(page_title="Ask RAG - USAT", layout="wide")
st.title("Ask RAG - Multi-file Support")
st.caption("Powered by Groq & LangChain | Asistente inteligente para documentos USAT")

# Upload multiple files
uploaded_files = st.file_uploader(
    "Upload files (PDF, DOCX, TXT, CSV)", 
    type=["pdf", "docx", "txt", "csv"], 
    accept_multiple_files=True,
    help="Sube uno o varios documentos para hacer preguntas sobre su contenido"
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
    
    # Display suggested questions
    if hasattr(st.session_state, 'suggested_questions') and st.session_state.suggested_questions:
        st.markdown("### 💡 Preguntas sugeridas:")
        cols = st.columns(len(st.session_state.suggested_questions))
        for idx, (col, question) in enumerate(zip(cols, st.session_state.suggested_questions)):
            with col:
                if st.button(f"❓ {question[:50]}...", key=f"suggestion_{idx}", use_container_width=True):
                    # Set the question as if user typed it
                    st.session_state.clicked_question = question

    # Display previous chat messages
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])
    
    # Export conversation functionality
    if st.session_state.messages:
        col1, col2 = st.columns([4, 1])
        with col2:
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

    # User input
    prompt = st.chat_input("Escribe tu pregunta aquí...")
    
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
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
else:
    st.info("Por favor, sube archivos para comenzar a hacer preguntas.")
    st.markdown("""
    ### 📖 Cómo usar:
    1. Sube uno o más documentos (PDF, DOCX, TXT, CSV)
    2. Espera a que se procesen
    3. Haz preguntas sobre el contenido
    4. Recibe respuestas basadas únicamente en los documentos
    """)

# Button to clear all messages and reset file upload
col1, col2 = st.columns([3, 1])
with col1:
    if st.button("🗑️ Limpiar conversación", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.session_state.first_interaction = True
        if 'suggested_questions' in st.session_state:
            del st.session_state.suggested_questions
        st.rerun()

with col2:
    if st.button("💾 Limpiar caché", type="secondary", use_container_width=True, help="Elimina archivos cacheados para liberar espacio"):
        cache_dir = Path("./vectorstore_cache")
        if cache_dir.exists():
            import shutil
            try:
                shutil.rmtree(cache_dir)
                st.success("✅ Caché limpiado")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        st.rerun()

# Delete temporary files on cleanup
if temp_files:
    for file_path in temp_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file {file_path}: {e}")
