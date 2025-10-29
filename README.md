# 🤖 RAG Multi-File Q&A System - Usatín

![GitHub Repo stars](https://img.shields.io/github/stars/Uni-Creator/RAG-MultiFile-QA?style=social) ![GitHub forks](https://img.shields.io/github/forks/Uni-Creator/RAG-MultiFile-QA?style=social)

> **Sistema inteligente de preguntas y respuestas basado en documentos utilizando Retrieval-Augmented Generation (RAG)**

Un asistente virtual amable y carismático llamado **Usatín** que responde preguntas basándose **únicamente** en el contenido de los documentos que subas. Perfecto para consultas sobre manuales, directivas, reglamentos y documentación institucional.

---

## 📋 Tabla de Contenidos

- [¿Qué es este proyecto?](#-qué-es-este-proyecto)
- [¿Cómo funciona?](#-cómo-funciona)
- [Características principales](#-características-principales)
- [Tecnologías utilizadas](#️-tecnologías-utilizadas)
- [Instalación](#-instalación)
- [Configuración](#️-configuración)
- [Uso](#-uso)
- [Arquitectura del sistema](#-arquitectura-del-sistema)
- [Estructura del proyecto](#-estructura-del-proyecto)
- [Preguntas frecuentes](#-preguntas-frecuentes)
- [Licencia](#-licencia)

---

## 🎯 ¿Qué es este proyecto?

**RAG Multi-File Q&A** es un sistema de inteligencia artificial que te permite:

1. **Subir múltiples documentos** (PDF, DOCX, TXT, CSV)
2. **Hacer preguntas** sobre el contenido de esos documentos
3. **Obtener respuestas precisas** basadas SOLO en la información de tus archivos
4. **Ver las fuentes** de donde proviene cada respuesta

**Usatín**, nuestro asistente virtual, utiliza tecnología RAG (Retrieval-Augmented Generation) para buscar información relevante en tus documentos y generar respuestas naturales y comprensibles.

---

## ⚙️ ¿Cómo funciona?

### Proceso paso a paso:

```
1. CARGA DE DOCUMENTOS
   ↓
   Usuario sube archivos (PDF, DOCX, TXT, CSV)
   ↓
2. PROCESAMIENTO
   ↓
   • Los documentos se dividen en fragmentos (chunks) de 1000 caracteres
   • Se crean embeddings (vectores) de cada fragmento
   • Se almacenan en una base de datos vectorial FAISS
   ↓
3. CONSULTA
   ↓
   Usuario hace una pregunta
   ↓
4. BÚSQUEDA
   ↓
   • La pregunta se convierte en un vector
   • Se buscan los 4 fragmentos más similares en FAISS
   • Se recuperan los fragmentos relevantes
   ↓
5. GENERACIÓN DE RESPUESTA
   ↓
   • Los fragmentos se envían como contexto a Groq API (Llama 3.1)
   • El LLM genera una respuesta basada SOLO en ese contexto
   • Se muestra la respuesta con las fuentes consultadas
   ↓
6. RESULTADO
   ↓
   Usuario recibe respuesta precisa con referencias
```

---

## ✨ Características principales

### 🎭 Asistente personalizado
- **Usatín** se presenta de forma amable en el primer mensaje
- Personalidad carismática y servicial
- Usa emojis para ser más expresivo
- Respuestas claras y bien estructuradas

### 📂 Soporte multi-formato
- ✅ **PDF** - Documentos de texto seleccionables
- ✅ **DOCX** - Archivos de Word
- ✅ **TXT** - Archivos de texto plano
- ✅ **CSV** - Datos tabulares

### 🔍 Búsqueda inteligente
- Embeddings normalizados para mayor precisión
- Búsqueda por similitud semántica
- Recuperación de múltiples fragmentos relevantes
- Identificación automática de fuentes

### 💬 Interfaz conversacional
- Chat interactivo estilo ChatGPT
- Historial de conversación
- Indicadores visuales de procesamiento
- Fuentes expandibles para cada respuesta

### 🎯 Respuestas confiables
- Basadas **únicamente** en los documentos subidos
- No inventa información
- Indica claramente cuando no encuentra respuesta
- Respeta formato de tablas y listas

---

## 🛠️ Tecnologías utilizadas

### Framework principal
- **Python 3.8+** - Lenguaje de programación
- **Streamlit** - Framework para la interfaz web interactiva

### Procesamiento de lenguaje natural
- **LangChain** - Framework para aplicaciones con LLMs
  - `langchain-community` - Loaders de documentos
  - `langchain-classic` - Cadenas de procesamiento
  - `langchain-core` - Componentes base
  - `langchain-huggingface` - Embeddings
  - `langchain-groq` - Integración con Groq API

### Modelo de lenguaje
- **Groq API** - Inferencia rápida de LLMs
- **Llama 3.1 8B Instant** - Modelo de lenguaje base
  - Temperature: 0.3 (respuestas precisas)
  - Max tokens: 1024 (respuestas completas)

### Embeddings y búsqueda
- **HuggingFace Embeddings** - all-MiniLM-L6-v2
  - Modelo ligero y eficiente
  - 384 dimensiones
  - Normalización activada
- **FAISS** - Base de datos vectorial de Facebook
  - Búsqueda ultra-rápida
  - Escalable a millones de documentos

### Procesamiento de documentos
- **PyPDF** - Lectura de archivos PDF
- **Unstructured** - Procesamiento de DOCX
- **Python-dotenv** - Gestión de variables de entorno
- **CSV Loader** - Procesamiento de archivos CSV

---

## 📥 Instalación

### Prerequisitos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Git

### Paso 1: Clonar el repositorio

```bash
git clone https://github.com/Uni-Creator/RAG-MultiFile-QA.git
cd RAG-MultiFile-QA
```

### Paso 2: Crear entorno virtual (recomendado)

```bash
# En macOS/Linux
python3 -m venv venv
source venv/bin/activate

# En Windows
python -m venv venv
venv\Scripts\activate
```

### Paso 3: Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## ⚙️ Configuración

### Obtener API Key de Groq

1. Ve a [console.groq.com](https://console.groq.com)
2. Crea una cuenta gratuita
3. Genera una API Key
4. Copia la clave

### Configurar variables de entorno

**Opción 1: Variable de entorno temporal (recomendado para pruebas)**

```bash
# En macOS/Linux
export GROQ_API_KEY="tu_api_key_aqui"

# En Windows (CMD)
set GROQ_API_KEY=tu_api_key_aqui

# En Windows (PowerShell)
$env:GROQ_API_KEY="tu_api_key_aqui"
```

**Opción 2: Archivo .env (para producción)**

```bash
# Crear archivo .env en la raíz del proyecto
echo "GROQ_API_KEY=tu_api_key_aqui" > .env
```

---

## 🚀 Uso

### Iniciar la aplicación

```bash
# Activar entorno virtual
source venv/bin/activate  # macOS/Linux
# o
venv\Scripts\activate     # Windows

# Configurar API Key
export GROQ_API_KEY="tu_api_key_aqui"

# Ejecutar aplicación
streamlit run main.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

### Usar el sistema

1. **Subir documentos**
   - Haz clic en "📂 Upload files"
   - Selecciona uno o varios archivos (PDF, DOCX, TXT, CSV)
   - Espera a que se procesen (verás "✅ X archivo(s) cargado(s) correctamente!")

2. **Hacer preguntas**
   - Escribe tu pregunta en el chat
   - Presiona Enter o haz clic en el botón de enviar
   - Usatín te saludará en el primer mensaje
   - Recibirás una respuesta basada en tus documentos

3. **Ver fuentes**
   - Expande "📚 Fuentes consultadas" para ver qué archivos se usaron
   - Verás el nombre de cada documento consultado

4. **Limpiar y reiniciar**
   - Haz clic en "🗑️ Limpiar todo" para empezar de nuevo
   - Esto borra el chat y permite subir nuevos documentos

### Ejemplos de preguntas

```
✅ Buenos ejemplos:
- "¿Cuáles son los requisitos para obtener una beca?"
- "¿Qué dice el reglamento sobre las actividades de formación complementaria?"
- "Resume la sección de responsabilidades"
- "¿Cuántas becas socioeconómicas se otorgan al año?"

❌ Malos ejemplos:
- "¿Qué opinas sobre...?" (Usatín solo usa los documentos)
- "Háblame de ti" (Preguntas fuera del contexto de los documentos)
```

---

## 🏗️ Arquitectura del sistema

```
┌─────────────────────────────────────────────────────────────┐
│                    INTERFAZ STREAMLIT                       │
│  (Usuario sube archivos y hace preguntas)                  │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              PROCESAMIENTO DE DOCUMENTOS                    │
│  • PyPDFLoader (PDF)                                        │
│  • UnstructuredWordDocumentLoader (DOCX)                    │
│  • TextLoader (TXT)                                         │
│  • CSVLoader (CSV)                                          │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              TEXT SPLITTING (Fragmentación)                 │
│  RecursiveCharacterTextSplitter                            │
│  • Chunk size: 1000 caracteres                             │
│  • Overlap: 200 caracteres                                 │
│  • Separadores inteligentes: \n\n, \n, ., espacio         │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              EMBEDDINGS (Vectorización)                     │
│  HuggingFace: all-MiniLM-L6-v2                             │
│  • 384 dimensiones                                          │
│  • Normalización activada                                  │
│  • Cada fragmento → Vector numérico                        │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              VECTOR STORE (FAISS)                          │
│  Base de datos vectorial en memoria                        │
│  • Índice optimizado para búsqueda rápida                 │
│  • Almacena vectores + metadatos                           │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              RETRIEVAL (Búsqueda)                          │
│  • Pregunta del usuario → Vector                           │
│  • Búsqueda de similitud en FAISS                          │
│  • Recupera top 4 fragmentos más relevantes               │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              PROMPT ENGINEERING                            │
│  • Contexto: Fragmentos recuperados                        │
│  • Pregunta: Query del usuario                             │
│  • Instrucciones: Personalidad de Usatín                  │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              LLM (Groq API - Llama 3.1)                    │
│  • Genera respuesta basada en contexto                     │
│  • Temperature: 0.3 (preciso)                              │
│  • Max tokens: 1024                                        │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              RESPUESTA AL USUARIO                          │
│  • Respuesta en lenguaje natural                          │
│  • Fuentes consultadas                                     │
│  • Formato amigable con emojis                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Estructura del proyecto

```
RAG-MultiFile-QA/
│
├── main.py                 # Aplicación principal de Streamlit
├── requirements.txt        # Dependencias del proyecto
├── README.md              # Este archivo
├── LICENSE                # Licencia MIT
│
├── venv/                  # Entorno virtual (no se sube a Git)
│   └── ...
│
└── .git/                  # Control de versiones
    └── ...
```

### Descripción de archivos clave

**main.py** - Aplicación principal con:
- Configuración de Streamlit
- Carga y procesamiento de documentos
- Sistema de embeddings y vectorstore
- Interfaz de chat
- Lógica de RAG con Groq API

**requirements.txt** - Todas las dependencias:
```
streamlit
langchain
langchain-community
langchain-classic
langchain-core
langchain-huggingface
langchain-groq
faiss-cpu
sentence-transformers
pypdf
python-docx
unstructured
python-dotenv
```

---

## ❓ Preguntas frecuentes

### ¿Puedo usar otros modelos de lenguaje?

Sí, el código está preparado para cambiar fácilmente el LLM. Actualmente usa Groq API con Llama 3.1, pero puedes modificar la configuración para usar:
- OpenAI GPT
- Anthropic Claude
- Otros modelos de Groq
- Modelos locales con Ollama

### ¿Los documentos se guardan en algún lado?

No. Los documentos se procesan en memoria temporal y se eliminan al cerrar la sesión o hacer clic en "Limpiar todo". Tu información es privada y no se almacena.

### ¿Cuántos documentos puedo subir a la vez?

No hay un límite técnico estricto, pero se recomienda:
- Hasta 10 documentos para rendimiento óptimo
- Tamaño total menor a 50MB
- Si subes muchos documentos, el procesamiento puede tardar más

### ¿Qué tan preciso es el sistema?

La precisión depende de:
- **Calidad de los documentos**: Texto claro y bien estructurado
- **Relevancia de la pregunta**: Preguntas específicas obtienen mejores resultados
- **Contenido disponible**: Solo responde con información de los documentos subidos

### ¿Funciona sin conexión a internet?

No. Requiere conexión para:
- Descargar el modelo de embeddings (primera vez)
- Hacer llamadas a Groq API
- Cargar la interfaz de Streamlit

### ¿Tiene límites la API de Groq?

Sí, la versión gratuita tiene límites de:
- Requests por minuto
- Tokens por día
- Para uso intensivo, considera un plan de pago

### ¿Puedo usarlo con documentos en otros idiomas?

Sí, el sistema funciona con múltiples idiomas. Sin embargo, está optimizado para español. Los embeddings y el LLM soportan:
- Español
- Inglés
- Otros idiomas con menor precisión

### ¿Cómo mejoro la calidad de las respuestas?

1. **Documentos bien estructurados** con títulos claros
2. **Preguntas específicas** en lugar de generales
3. **Nombres de archivo descriptivos**
4. **Evitar PDFs escaneados** (usar PDFs con texto seleccionable)
5. **Documentos relevantes** al tema de consulta

---

## 📜 Licencia

Este proyecto está bajo la **Licencia MIT**.

```
MIT License

Copyright (c) 2025 Uni-Creator

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Si encuentras un bug o tienes una sugerencia:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

---

## 📧 Contacto

Para preguntas o soporte:
- GitHub: [@Uni-Creator](https://github.com/Uni-Creator)
- Issues: [Reportar un problema](https://github.com/Uni-Creator/RAG-MultiFile-QA/issues)

---

## 🙏 Agradecimientos

- **Groq** por su API rápida y eficiente
- **LangChain** por el framework de RAG
- **HuggingFace** por los modelos de embeddings
- **Streamlit** por la interfaz web fácil de usar
- **Meta** por FAISS y Llama

---

<div align="center">

### ⭐ Si te gusta este proyecto, dale una estrella en GitHub! ⭐

**Hecho con ❤️ por la comunidad**

</div>

