#!/usr/bin/env python3
"""
Script para procesar documentos desde la carpeta local 'documents/'
Coloca tus archivos PDF, DOCX, TXT, CSV en la carpeta 'documents/' y ejecuta este script.
"""

import os
import sys
import hashlib
import pickle
from pathlib import Path

# Añadir el directorio actual al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Imports del sistema RAG
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader, CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

DOCUMENTS_FOLDER = './documents'
CACHE_FOLDER = './vectorstore_cache'
ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.txt', '.csv'}

def get_documents_hash():
    """Genera un hash basado en los archivos en la carpeta documents"""
    hash_content = ""
    doc_path = Path(DOCUMENTS_FOLDER)
    
    if not doc_path.exists():
        print(f"❌ La carpeta '{DOCUMENTS_FOLDER}' no existe.")
        return None
    
    files = sorted([f for f in doc_path.iterdir() if f.suffix.lower() in ALLOWED_EXTENSIONS])
    
    if not files:
        print(f"❌ No se encontraron archivos en '{DOCUMENTS_FOLDER}'")
        print(f"   Formatos permitidos: {', '.join(ALLOWED_EXTENSIONS)}")
        return None
    
    print(f"\n📁 Archivos encontrados:")
    for file in files:
        print(f"   • {file.name}")
        with open(file, 'rb') as f:
            content = f.read()
            hash_content += f"{file.name}_{hashlib.md5(content).hexdigest()}"
    
    return hashlib.md5(hash_content.encode()).hexdigest()

def load_local_documents():
    """Carga y procesa documentos desde la carpeta local"""
    print("\n🔍 Procesando documentos...")
    
    # Verificar si ya existe en caché
    files_hash = get_documents_hash()
    if not files_hash:
        return False
    
    cache_path = Path(CACHE_FOLDER) / f"{files_hash}.pkl"
    
    if cache_path.exists():
        print(f"\n✅ Vectorstore encontrado en caché: {files_hash[:8]}...")
        print("   No es necesario reprocesar.")
        return True
    
    # Cargar documentos
    print("\n⚙️  Cargando documentos...")
    doc_path = Path(DOCUMENTS_FOLDER)
    all_documents = []
    
    for file_path in doc_path.iterdir():
        if file_path.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue
        
        print(f"   Procesando: {file_path.name}...")
        
        try:
            if file_path.suffix == '.pdf':
                loader = PyPDFLoader(str(file_path))
            elif file_path.suffix == '.txt':
                loader = TextLoader(str(file_path), encoding='utf-8')
            elif file_path.suffix == '.docx':
                loader = UnstructuredWordDocumentLoader(str(file_path))
            elif file_path.suffix == '.csv':
                loader = CSVLoader(str(file_path))
            else:
                continue
            
            docs = loader.load()
            
            # Añadir metadata
            for i, doc in enumerate(docs):
                doc.metadata.update({
                    'source_file': file_path.name,
                    'file_type': file_path.suffix,
                    'chunk_index': i
                })
                if 'page' in doc.metadata:
                    doc.metadata['page_number'] = doc.metadata['page']
            
            all_documents.extend(docs)
            print(f"      ✓ {len(docs)} documento(s) cargado(s)")
            
        except Exception as e:
            print(f"      ✗ Error: {str(e)}")
            continue
    
    if not all_documents:
        print("\n❌ No se pudieron cargar documentos.")
        return False
    
    print(f"\n📄 Total de documentos cargados: {len(all_documents)}")
    
    # Dividir en chunks
    print("\n✂️  Dividiendo en chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(all_documents)
    print(f"   ✓ {len(chunks)} chunks creados")
    
    # Crear embeddings
    print("\n🧠 Generando embeddings...")
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Crear vectorstore
    print("   Creando vectorstore (esto puede tardar)...")
    vectorstore = FAISS.from_documents(chunks, embeddings_model)
    
    # Guardar en caché
    print(f"\n💾 Guardando en caché: {files_hash[:8]}...")
    os.makedirs(CACHE_FOLDER, exist_ok=True)
    
    with open(cache_path, 'wb') as f:
        pickle.dump({
            'vectorstore': vectorstore,
            'files_hash': files_hash,
            'num_documents': len(all_documents),
            'num_chunks': len(chunks)
        }, f)
    
    print("\n✅ ¡Documentos procesados exitosamente!")
    print(f"   • Documentos: {len(all_documents)}")
    print(f"   • Chunks: {len(chunks)}")
    print(f"   • Hash: {files_hash[:8]}")
    
    return True

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Procesador de Documentos Locales - RAG System")
    print("=" * 60)
    
    # Crear carpeta documents si no existe
    os.makedirs(DOCUMENTS_FOLDER, exist_ok=True)
    
    success = load_local_documents()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ Proceso completado. Puedes iniciar el servidor Flask.")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("❌ Error en el procesamiento.")
        print("=" * 60)
        sys.exit(1)
