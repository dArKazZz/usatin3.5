from flask import Flask, render_template, request, jsonify, session, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import hashlib
import pickle
from datetime import datetime
from pathlib import Path
from io import BytesIO
import tempfile
import re
import warnings
import logging

# Silenciar warnings
warnings.filterwarnings('ignore', category=UserWarning, module='urllib3')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Configurar logging para Flask
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Langchain imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, CSVLoader
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'usatin-secret-key-2025'
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max
CORS(app)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('./vectorstore_cache', exist_ok=True)
os.makedirs('./documents', exist_ok=True)

# Load API key
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

# Setup LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    groq_api_key=api_key,
    temperature=0.3,
    max_tokens=2048
)

# Global variables
vectorstore = None
embeddings_model = None
qa_chain = None

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'csv'}
DOCUMENTS_FOLDER = './documents'

def load_local_documents_on_startup():
    """Carga documentos desde la carpeta local al iniciar el servidor"""
    global vectorstore, qa_chain, embeddings_model
    
    doc_path = Path(DOCUMENTS_FOLDER)
    if not doc_path.exists():
        print(f"⚠️  Carpeta '{DOCUMENTS_FOLDER}' no existe. Créala y coloca archivos allí.")
        return
    
    files = list(doc_path.glob('*'))
    valid_files = [f for f in files if f.suffix.lower() in {'.pdf', '.docx', '.txt', '.csv'}]
    
    if not valid_files:
        print(f"⚠️  No se encontraron archivos en '{DOCUMENTS_FOLDER}'")
        return
    
    print(f"📁 Cargando {len(valid_files)} documento(s)...")
    
    # Generar hash de los archivos
    hash_content = ""
    for file in sorted(valid_files):
        with open(file, 'rb') as f:
            content = f.read()
            hash_content += f"{file.name}_{hashlib.md5(content).hexdigest()}"
    
    files_hash = hashlib.md5(hash_content.encode()).hexdigest()
    cache_path = Path("./vectorstore_cache") / f"{files_hash}.pkl"
    
    # Intentar cargar desde caché
    if cache_path.exists():
        try:
            print("✅ Cargando vectorstore desde caché...")
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)
                vectorstore = cached_data['vectorstore']
                qa_chain = create_qa_chain(vectorstore)
                print(f"✅ Sistema listo - {cached_data.get('num_documents', '?')} documentos, {cached_data.get('num_chunks', '?')} chunks")
                return
        except Exception as e:
            print(f"✗ Error al cargar caché: {e}")
    
    print("⚠️  No se encontró caché. Ejecuta: python process_local_documents.py")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_files_hash(files_data):
    """Generate hash based on file contents"""
    hash_content = ""
    for filename, content in files_data.items():
        hash_content += f"{filename}_{hashlib.md5(content).hexdigest()}"
    return hashlib.md5(hash_content.encode()).hexdigest()

def load_documents(files_data, files_hash):
    """Load and process documents"""
    cache_dir = Path("./vectorstore_cache")
    cache_path = cache_dir / f"{files_hash}.pkl"
    
    # Try to load from cache
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)
                return cached_data['vectorstore'], None
        except Exception as e:
            print(f"Cache load error: {e}")
    
    # Process documents
    loaders = []
    temp_files = []
    
    for filename, content in files_data.items():
        # Save to temp file
        suffix = os.path.splitext(filename)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name
            temp_files.append(temp_path)
        
        # Create appropriate loader
        if filename.endswith('.pdf'):
            loaders.append(PyPDFLoader(temp_path))
        elif filename.endswith('.txt'):
            loaders.append(TextLoader(temp_path))
        elif filename.endswith('.docx'):
            loaders.append(UnstructuredWordDocumentLoader(temp_path))
        elif filename.endswith('.csv'):
            loaders.append(CSVLoader(temp_path))
    
    # Load documents
    documents = []
    for idx, loader in enumerate(loaders):
        try:
            docs = loader.load()
            for doc_idx, doc in enumerate(docs):
                doc.metadata['source_file'] = list(files_data.keys())[idx]
                doc.metadata['file_index'] = idx
                doc.metadata['doc_index'] = doc_idx
                doc.metadata['upload_time'] = datetime.now().isoformat()
                if 'page' in doc.metadata:
                    doc.metadata['page_number'] = doc.metadata['page'] + 1
            documents.extend(docs)
        except Exception as e:
            print(f"Error loading document: {e}")
    
    if not documents:
        return None, "No documents could be loaded"
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
        length_function=len,
    )
    texts = text_splitter.split_documents(documents)
    
    # Add chunk metadata
    for idx, text in enumerate(texts):
        text.metadata['chunk_index'] = idx
        text.metadata['chunk_length'] = len(text.page_content)
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    # Save to cache
    try:
        cache_data = {
            'vectorstore': vectorstore,
            'created_at': datetime.now().isoformat(),
            'num_docs': len(documents),
            'num_chunks': len(texts)
        }
        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f)
    except Exception as e:
        print(f"Cache save error: {e}")
    
    # Cleanup temp files
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
        except:
            pass
    
    return vectorstore, None

def create_qa_chain(vectorstore):
    """Create QA chain with custom prompt"""
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
        2. Si no encuentras la respuesta, di amablemente: "Lo siento, esa información no está disponible en los documentos. ¿Hay algo más en lo que pueda ayudarte?"
        3. Si encuentras información parcial, indícalo con empatía
        4. Mantén las respuestas claras, bien estructuradas y fáciles de entender
        5. Termina tus respuestas invitando a hacer más preguntas cuando sea apropiado

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
            search_kwargs={"k": 10}
        ),
        input_key="question",
        return_source_documents=True,
        chain_type_kwargs={"prompt": custom_prompt}
    )
    
    return chain

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file upload"""
    global vectorstore, qa_chain
    
    # Accept both 'files' and 'files[]'
    if 'files' in request.files:
        files = request.files.getlist('files')
    elif 'files[]' in request.files:
        files = request.files.getlist('files[]')
    else:
        return jsonify({'error': 'No files provided'}), 400
    
    if not files or files[0].filename == '':
        return jsonify({'error': 'No files selected'}), 400
    
    # Read files into memory
    files_data = {}
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            content = file.read()
            files_data[filename] = content
    
    if not files_data:
        return jsonify({'error': 'No valid files uploaded'}), 400
    
    # Generate hash and load documents
    files_hash = get_files_hash(files_data)
    vectorstore, error = load_documents(files_data, files_hash)
    
    if error:
        return jsonify({'error': error}), 500
    
    # Create QA chain
    qa_chain = create_qa_chain(vectorstore)
    
    # Store in session
    session['files_loaded'] = True
    session['num_files'] = len(files_data)
    session['file_names'] = list(files_data.keys())
    
    return jsonify({
        'success': True,
        'message': f'{len(files_data)} archivo(s) procesado(s) correctamente',
        'files': list(files_data.keys())
    })

@app.route('/ask', methods=['POST'])
def ask_question():
    """Handle question"""
    global qa_chain
    
    if not qa_chain:
        return jsonify({'error': 'Por favor sube documentos primero'}), 400
    
    data = request.json
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({'error': 'Pregunta vacía'}), 400
    
    try:
        # Process question
        response = qa_chain.invoke({"question": question})
        answer = response.get("result", "No se pudo generar respuesta")
        source_docs = response.get("source_documents", [])
        
        # Format sources
        sources = []
        for doc in source_docs[:5]:
            sources.append({
                'file': doc.metadata.get('source_file', 'Desconocido'),
                'page': doc.metadata.get('page_number', 'N/A'),
                'chunk': doc.metadata.get('chunk_index', 'N/A'),
                'preview': doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
            })
        
        return jsonify({
            'success': True,
            'answer': answer,
            'sources': sources
        })
    
    except Exception as e:
        return jsonify({'error': f'Error al procesar pregunta: {str(e)}'}), 500

@app.route('/clear-cache', methods=['POST'])
def clear_cache():
    """Clear vectorstore cache"""
    try:
        cache_dir = Path("./vectorstore_cache")
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
            os.makedirs('./vectorstore_cache', exist_ok=True)
        return jsonify({'success': True, 'message': 'Caché limpiado'})
    except Exception as e:
        return jsonify({'error': f'Error limpiando caché: {str(e)}'}), 500

@app.route('/status', methods=['GET'])
def status():
    """Get app status"""
    return jsonify({
        'files_loaded': session.get('files_loaded', False),
        'num_files': session.get('num_files', 0),
        'file_names': session.get('file_names', [])
    })

if __name__ == '__main__':
    print("\n🚀 Iniciando servidor RAG...")
    
    # Cargar documentos locales al inicio
    load_local_documents_on_startup()
    
    print("✅ Servidor iniciado en http://localhost:5001\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
