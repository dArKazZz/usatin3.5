from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import hashlib
import pickle
from datetime import datetime
from pathlib import Path
import tempfile
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
from sentence_transformers import CrossEncoder
import numpy as np

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Ensure folders exist
os.makedirs('./vectorstore_cache', exist_ok=True)
os.makedirs('./documents', exist_ok=True)

# Load API key
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

# Setup LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=api_key,
    temperature=0.3,
    max_tokens=2048
)

# Global variables
vectorstore = None
qa_chain = None
reranker_model = None

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

def initialize_reranker():
    """Inicializar modelo de reranking"""
    global reranker_model
    if reranker_model is None:
        try:
            print("📊 Cargando modelo de reranking...")
            reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
            print("✅ Reranker listo")
        except Exception as e:
            print(f"⚠️  Error cargando reranker: {e}")
            print("   Continuando sin reranking...")
            reranker_model = None

def rerank_documents(query, documents, top_k=5):
    """Re-ordenar documentos por relevancia usando reranker"""
    global reranker_model
    
    if reranker_model is None or not documents:
        return documents[:top_k]
    
    try:
        # Crear pares query-document
        pairs = [[query, doc.page_content[:512]] for doc in documents]  # Limitar a 512 tokens
        
        # Obtener scores
        scores = reranker_model.predict(pairs)
        
        # Ordenar por score descendente
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Retornar top_k documentos
        return [doc for doc, score in scored_docs[:top_k]]
    except Exception as e:
        print(f"⚠️  Error en reranking: {e}")
        return documents[:top_k]

def rephrase_query(original_query):
    """Reformular la consulta para mejorar la búsqueda"""
    try:
        rephrase_prompt = f"""Reformula la siguiente pregunta para que sea más clara y específica, manteniendo el mismo significado. 
Si la pregunta ya es clara, devuélvela sin cambios.

Pregunta original: {original_query}

Pregunta reformulada (solo devuelve la pregunta, sin explicaciones adicionales):"""
        
        response = llm.invoke(rephrase_prompt)
        rephrased = response.content.strip()
        
        # Si la reformulación es muy diferente o muy larga, usar la original
        if len(rephrased) > len(original_query) * 2 or len(rephrased) < 5:
            return original_query
        
        return rephrased
    except Exception as e:
        print(f"⚠️  Error en query rephrasing: {e}")
        return original_query

def format_citations(documents):
    """Formatear citas de las fuentes"""
    citations = []
    seen = set()
    
    for doc in documents:
        source_file = doc.metadata.get('source_file', 'Desconocido')
        page = doc.metadata.get('page_number', doc.metadata.get('page', 'N/A'))
        
        # Crear identificador único para evitar duplicados
        citation_id = f"{source_file}_p{page}"
        
        if citation_id not in seen:
            seen.add(citation_id)
            citations.append({
                'file': source_file,
                'page': page,
                'preview': doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            })
    
    return citations

def create_qa_chain(vectorstore):
    """Create QA chain with custom prompt"""
    custom_prompt = PromptTemplate(
        template="""
        Tu nombre es Usatín, un asistente virtual amable, carismático y servicial de la Universidad USAT.
        
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
            search_kwargs={"k": 15}  # Aumentado para reranking
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

@app.route('/ask', methods=['POST'])
def ask_question():
    """Handle question with reranking, rephrasing and citations"""
    global qa_chain, vectorstore
    
    if not qa_chain:
        return jsonify({'error': 'Por favor sube documentos primero'}), 400
    
    data = request.json
    question = data.get('question', '').strip()
    
    if not question:
        return jsonify({'error': 'Pregunta vacía'}), 400
    
    try:
        # 1. QUERY REPHRASING - Reformular consulta
        rephrased_question = rephrase_query(question)
        use_rephrased = rephrased_question != question
        
        # 2. RETRIEVAL - Obtener documentos candidatos
        search_query = rephrased_question if use_rephrased else question
        initial_docs = vectorstore.similarity_search(search_query, k=15)
        
        # 3. RERANKING - Re-ordenar por relevancia
        reranked_docs = rerank_documents(question, initial_docs, top_k=5)
        
        # 4. GENERATION - Generar respuesta con documentos re-ordenados
        # Crear contexto con los documentos rerankeados
        context = "\n\n".join([doc.page_content for doc in reranked_docs])
        
        custom_prompt = f"""
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
        """
        
        response = llm.invoke(custom_prompt)
        answer = response.content
        
        # 5. CITATIONS - Formatear citas de las fuentes
        citations = format_citations(reranked_docs)
        
        return jsonify({
            'success': True,
            'answer': answer,
            'sources': citations,
            'query_rephrased': use_rephrased,
            'original_query': question if use_rephrased else None,
            'rephrased_query': rephrased_question if use_rephrased else None
        })
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error al procesar pregunta: {str(e)}'}), 500

if __name__ == '__main__':
    print("\n🚀 Iniciando servidor RAG...")
    
    # Inicializar reranker
    initialize_reranker()
    
    # Cargar documentos locales al inicio
    load_local_documents_on_startup()
    
    print("✅ Servidor iniciado en http://localhost:5001\n")
    
    app.run(debug=True, host='127.0.0.1', port=5001)
