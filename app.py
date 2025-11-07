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

# Langchain imports
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, CSVLoader
from langchain_classic.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# Voice imports
try:
    import speech_recognition as sr
    from gtts import gTTS
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False

# Set ffmpeg path for pydub
import subprocess
ffmpeg_paths = [
    '/opt/homebrew/bin/ffmpeg',
    '/usr/local/bin/ffmpeg',
    '/usr/bin/ffmpeg'
]
for path in ffmpeg_paths:
    if os.path.exists(path):
        os.environ['PATH'] = os.path.dirname(path) + ':' + os.environ.get('PATH', '')
        break

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
    model="llama-3.1-8b-instant",
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
    global vectorstore, qa_chain
    
    doc_path = Path(DOCUMENTS_FOLDER)
    if not doc_path.exists():
        print(f"⚠️  Carpeta '{DOCUMENTS_FOLDER}' no existe. Créala y coloca archivos allí.")
        return
    
    files = list(doc_path.glob('*'))
    valid_files = [f for f in files if f.suffix.lower() in {'.pdf', '.docx', '.txt', '.csv'}]
    
    if not valid_files:
        print(f"⚠️  No se encontraron archivos en '{DOCUMENTS_FOLDER}'")
        print(f"   Coloca archivos PDF, DOCX, TXT o CSV en esa carpeta.")
        return
    
    print(f"\n📁 Archivos encontrados en {DOCUMENTS_FOLDER}:")
    for file in valid_files:
        print(f"   • {file.name}")
    
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
            print(f"\n✅ Cargando vectorstore desde caché...")
            with open(cache_path, "rb") as f:
                cached_data = pickle.load(f)
                vectorstore = cached_data['vectorstore']
                qa_chain = create_qa_chain(vectorstore)
                print(f"   ✓ Vectorstore cargado correctamente")
                print(f"   • Documentos: {cached_data.get('num_documents', '?')}")
                print(f"   • Chunks: {cached_data.get('num_chunks', '?')}")
                return
        except Exception as e:
            print(f"   ✗ Error al cargar caché: {e}")
    
    print(f"\n⚠️  No se encontró caché para estos documentos.")
    print(f"   Ejecuta primero: python process_local_documents.py")

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

def clean_text_for_speech(text):
    """Remove markdown and emojis from text for TTS"""
    clean_text = text
    
    # Remove markdown formatting
    clean_text = re.sub(r'\*\*\*(.+?)\*\*\*', r'\1', clean_text)
    clean_text = re.sub(r'\*\*(.+?)\*\*', r'\1', clean_text)
    clean_text = re.sub(r'\*(.+?)\*', r'\1', clean_text)
    clean_text = re.sub(r'___(.+?)___', r'\1', clean_text)
    clean_text = re.sub(r'__(.+?)__', r'\1', clean_text)
    clean_text = re.sub(r'_(.+?)_', r'\1', clean_text)
    clean_text = clean_text.replace('*', '').replace('_', '')
    clean_text = re.sub(r'^#{1,6}\s+', '', clean_text, flags=re.MULTILINE)
    clean_text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', clean_text)
    clean_text = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', r'\1', clean_text)
    clean_text = re.sub(r'`([^`]+)`', r'\1', clean_text)
    clean_text = re.sub(r'```[^\n]*\n(.*?)```', r'\1', clean_text, flags=re.DOTALL)
    clean_text = re.sub(r'^[\-_\*]{3,}$', '', clean_text, flags=re.MULTILINE)
    clean_text = re.sub(r'^\s*[\-\*\+]\s+', '', clean_text, flags=re.MULTILINE)
    clean_text = re.sub(r'^\s*\d+\.\s+', '', clean_text, flags=re.MULTILINE)
    clean_text = re.sub(r'^\s*>\s+', '', clean_text, flags=re.MULTILINE)
    
    # Remove emojis
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U00002600-\U000026FF"
        "\U00002B50"
        "\U0001F004"
        "]+", 
        flags=re.UNICODE
    )
    clean_text = emoji_pattern.sub('', clean_text)
    clean_text = re.sub(r'\n\s*\n', '\n\n', clean_text)
    clean_text = clean_text.strip()
    
    return clean_text

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html', voice_available=VOICE_AVAILABLE)

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

@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    """Convert speech to text"""
    if not VOICE_AVAILABLE:
        return jsonify({'error': 'Funcionalidad de voz no disponible'}), 400
    
    if 'audio' not in request.files:
        return jsonify({'error': 'No se recibió archivo de audio'}), 400
    
    audio_file = request.files['audio']
    
    if audio_file.filename == '':
        return jsonify({'error': 'Archivo de audio vacío'}), 400
    
    temp_webm = None
    temp_wav = None
    
    try:
        print("=== Iniciando reconocimiento de voz ===")
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 300
        recognizer.dynamic_energy_threshold = True
        
        # Save webm file
        temp_webm = tempfile.mktemp(suffix='.webm')
        audio_file.save(temp_webm)
        print(f"Audio guardado en: {temp_webm}")
        
        # Check file size
        file_size = os.path.getsize(temp_webm)
        print(f"Tamaño del archivo: {file_size} bytes")
        
        if file_size < 1000:  # Less than 1KB
            return jsonify({'error': 'El audio es demasiado corto. Intenta hablar más tiempo.'}), 400
        
        # Try to convert to WAV using pydub
        try:
            print("Importando pydub...")
            from pydub import AudioSegment
            
            # Set ffmpeg path explicitly
            AudioSegment.converter = "/opt/homebrew/bin/ffmpeg"
            AudioSegment.ffprobe = "/opt/homebrew/bin/ffprobe"
            
            print("Cargando archivo de audio...")
            # Load audio file
            audio = AudioSegment.from_file(temp_webm, format='webm')
            
            print("Convirtiendo a mono 16kHz...")
            # Convert to mono and set sample rate
            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(16000)
            
            # Export as WAV
            temp_wav = tempfile.mktemp(suffix='.wav')
            print(f"Exportando a WAV: {temp_wav}")
            audio.export(temp_wav, format='wav')
            
            print("Reconociendo voz...")
            # Recognize speech
            with sr.AudioFile(temp_wav) as source:
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio_data = recognizer.record(source)
                print("Llamando a Google Speech Recognition...")
                text = recognizer.recognize_google(audio_data, language="es-ES")
            
            print(f"Texto reconocido: {text}")
            
        except ImportError as ie:
            print(f"Error de importación: {ie}")
            return jsonify({'error': 'Se requiere ffmpeg para el reconocimiento de voz. Por favor instálalo con: brew install ffmpeg'}), 500
        
        except sr.UnknownValueError:
            print("No se pudo entender el audio")
            return jsonify({'error': 'No se pudo entender el audio. Intenta hablar más claro y cerca del micrófono.'}), 400
        
        except sr.RequestError as e:
            print(f"Error de Google Speech API: {e}")
            return jsonify({'error': f'Error del servicio de Google: {str(e)}'}), 500
        
        except Exception as inner_e:
            print(f"Error interno: {inner_e}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'Error al procesar audio: {str(inner_e)}'}), 500
        
        return jsonify({
            'success': True,
            'text': text
        })
    
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"Error general en speech-to-text:\n{error_detail}")
        return jsonify({'error': f'Error al procesar audio: {str(e)}'}), 500
    
    finally:
        # Cleanup temp files
        if temp_webm and os.path.exists(temp_webm):
            try:
                os.remove(temp_webm)
                print(f"Archivo temporal eliminado: {temp_webm}")
            except:
                pass
        if temp_wav and os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
                print(f"Archivo temporal eliminado: {temp_wav}")
            except:
                pass

@app.route('/text-to-speech', methods=['POST'])
def text_to_speech():
    """Convert text to speech"""
    if not VOICE_AVAILABLE:
        return jsonify({'error': 'Funcionalidad de voz no disponible'}), 400
    
    data = request.json
    text = data.get('text', '').strip()
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        # Clean text
        clean_text = clean_text_for_speech(text)
        
        # Generate speech with gTTS (fast mode)
        tts = gTTS(text=clean_text, lang='es', slow=False)
        
        # Save to temporary file
        temp_mp3 = tempfile.mktemp(suffix='.mp3')
        tts.save(temp_mp3)
        
        # Speed up audio using pydub (1.3x faster)
        try:
            from pydub import AudioSegment
            from pydub.effects import speedup
            
            # Load audio
            audio = AudioSegment.from_mp3(temp_mp3)
            
            # Speed up by 1.3x (adjust this value: 1.2 = 20% faster, 1.5 = 50% faster)
            faster_audio = speedup(audio, playback_speed=1)
            
            # Export to bytes
            audio_bytes = BytesIO()
            faster_audio.export(audio_bytes, format='mp3')
            audio_bytes.seek(0)
            
            # Cleanup
            os.remove(temp_mp3)
            
            return send_file(audio_bytes, mimetype='audio/mp3', as_attachment=False)
            
        except ImportError:
            # If pydub not available, return normal speed
            audio_bytes = BytesIO()
            with open(temp_mp3, 'rb') as f:
                audio_bytes.write(f.read())
            audio_bytes.seek(0)
            os.remove(temp_mp3)
            return send_file(audio_bytes, mimetype='audio/mp3', as_attachment=False)
    
    except Exception as e:
        return jsonify({'error': f'Error generando audio: {str(e)}'}), 500

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
        'file_names': session.get('file_names', []),
        'voice_available': VOICE_AVAILABLE
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Iniciando servidor RAG con documentos locales")
    print("="*60)
    
    # Cargar documentos locales al inicio
    load_local_documents_on_startup()
    
    print("\n" + "="*60)
    print("✅ Servidor iniciado en: http://localhost:5001")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
