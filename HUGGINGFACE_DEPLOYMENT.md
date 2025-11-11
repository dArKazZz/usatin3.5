# 🤗 Deployment en Hugging Face Spaces

## 🚀 PASO A PASO

### 1. Crear cuenta en Hugging Face
- Ve a: https://huggingface.co/join
- Regístrate (gratis, no requiere tarjeta)

### 2. Crear un nuevo Space
- Ve a: https://huggingface.co/spaces
- Click en "Create new Space"
- Configura:
  ```
  Space name: chatbot-rag-usat (o el que prefieras)
  License: MIT
  Space SDK: Docker
  Space hardware: CPU basic (gratuito)
  ```
- Click "Create Space"

### 3. Subir archivos al Space

Opción A - Desde GitHub (RECOMENDADO):
```bash
# 1. Clonar el repo del Space
git clone https://huggingface.co/spaces/TU_USUARIO/chatbot-rag-usat
cd chatbot-rag-usat

# 2. Copiar archivos de tu proyecto
cp -r /ruta/a/usatin3.5/* .

# 3. Hacer commit y push
git add .
git commit -m "Initial commit"
git push
```

Opción B - Interfaz web:
1. En tu Space, ve a "Files and versions"
2. Click "Add file" → "Upload files"
3. Arrastra todos los archivos del proyecto
4. Click "Commit changes to main"

### 4. Configurar Secrets (Variables de entorno)
- En tu Space, ve a "Settings"
- Scroll hasta "Repository secrets"
- Click "New secret"
- Agrega:
  ```
  Name: GROQ_API_KEY
  Value: [TU_GROQ_API_KEY_AQUI]
  ```
  ⚠️ **Usa tu propia API key de Groq: https://console.groq.com**
- Click "Add"

### 5. Esperar el build
- Hugging Face detectará el Dockerfile
- Construirá la imagen (5-10 minutos)
- Iniciará automáticamente
- Verás logs en tiempo real

### 6. ¡Listo! Accede a tu chatbot
- URL: https://huggingface.co/spaces/TU_USUARIO/chatbot-rag-usat
- O comparte: https://TU_USUARIO-chatbot-rag-usat.hf.space

## 📋 ARCHIVOS INCLUIDOS PARA HUGGING FACE

✅ `README.md` - Configuración del Space
✅ `Dockerfile` - Imagen Docker
✅ `.dockerignore` - Archivos a ignorar
✅ `requirements.txt` - Dependencias Python
✅ `app.py` - Aplicación Flask (configurado para puerto 7860)
✅ `documents/` - PDFs procesados
✅ `vectorstore_cache/` - Cache pre-generado

## 🎯 VENTAJAS DE HUGGING FACE SPACES

✅ **100% Gratuito**
   - Sin límites de tiempo
   - Sin tarjeta de crédito
   - Para siempre

✅ **Optimizado para ML**
   - Hardware especializado
   - Librería de modelos integrada
   - Community de ML/AI

✅ **Siempre activo**
   - No se duerme
   - Respuesta inmediata
   - Alta disponibilidad

✅ **Fácil de compartir**
   - URL pública
   - Embebible en websites
   - Interfaz profesional

## 🔧 CONFIGURACIÓN OPCIONAL

### Mejorar el hardware (si necesitas más potencia):
1. Settings → Hardware
2. Cambiar a:
   - CPU upgrade (mejor CPU) - GRATIS
   - T4 GPU small (para inference ML rápido) - $0.60/hora
   - A10G GPU (para modelos grandes) - $3.15/hora

Para tu proyecto, **CPU basic es suficiente** (gratis).

### Hacer el Space privado:
1. Settings → Visibility
2. Cambiar a "Private"
3. Solo tú y usuarios autorizados podrán acceder

### Agregar colaboradores:
1. Settings → Members
2. Add member
3. Ingresar username de Hugging Face

## 📊 MONITOREO

En tu Space puedes ver:
- **Logs**: Pestaña "Logs"
- **Métricas**: Uso de CPU/RAM
- **Versiones**: Historial de cambios
- **Duplicaciones**: Cuántas personas han duplicado tu Space

## 🔄 ACTUALIZAR EL DEPLOYMENT

```bash
# 1. Hacer cambios en tu código local
# 2. Commit y push

cd /ruta/al/space
git add .
git commit -m "Update: descripción de cambios"
git push

# Hugging Face rebuildeará automáticamente
```

## ⚠️ TROUBLESHOOTING

### Build falla por timeout:
- Aumenta a CPU upgrade (gratis)
- Los modelos ML tardan en descargar

### Error de memoria:
- CPU basic tiene 16GB RAM (suficiente)
- Si aún falla, upgrade a CPU upgrade

### Modelo de reranking muy pesado:
- Comenta `initialize_reranker()` en app.py
- O usa CPU upgrade (gratis)

### API key no funciona:
- Verifica que esté en Secrets
- Nombre exacto: GROQ_API_KEY
- Reinicia el Space después de agregar

## 🎉 RESULTADO FINAL

Tu chatbot estará en:
```
https://TU_USUARIO-chatbot-rag-usat.hf.space
```

- ✅ 100% gratis
- ✅ Siempre activo
- ✅ Rápido y confiable
- ✅ Fácil de compartir

## 📱 COMPARTIR

Puedes:
1. Compartir URL directa
2. Embedear en tu web con iframe
3. Usar API de Hugging Face
4. Conectar con Gradio/Streamlit

## 🌟 TIPS

1. **Agrega un banner bonito** en README.md
2. **Escribe documentación clara** para usuarios
3. **Sube capturas** de pantalla del chatbot
4. **Comparte en la comunidad** de HF

¡Ya está todo listo! Solo sube los archivos a Hugging Face Spaces. 🚀
