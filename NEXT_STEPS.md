# ✅ PROYECTO SINCRONIZADO CON GITHUB

Tu proyecto está listo para deployment en Railway.

## 📊 Lo que se ha subido:

✅ Código fuente completo con mejoras FACTS:
   - Reranking con CrossEncoder
   - Query Rephrasing con LLM
   - Citations automáticas
   - Interfaz mejorada con soporte Markdown

✅ Configuración para Railway:
   - Procfile
   - runtime.txt (Python 3.9)
   - railway.toml
   - Variables de entorno preparadas

✅ Documentos y vectorstore:
   - 6 documentos PDF
   - Vectorstore pre-procesado
   - 80 documentos, 169 chunks

## 🚀 SIGUIENTE PASO: DEPLOYMENT EN RAILWAY

### 1. Ve a Railway
   URL: https://railway.app

### 2. Crear cuenta
   - Haz clic en "Login"
   - Selecciona "Login with GitHub"
   - Autoriza Railway

### 3. Crear nuevo proyecto
   - Clic en "New Project"
   - Selecciona "Deploy from GitHub repo"
   - Busca y selecciona: **dArKazZz/usatin3.5**
   - Clic en "Deploy Now"

### 4. Configurar variables de entorno
   Una vez creado el proyecto:
   - Ve a la pestaña "Variables"
   - Clic en "New Variable"
   - Agrega:
     ```
     Variable: GROQ_API_KEY
     Value: [TU_GROQ_API_KEY_AQUI]
     ```
   ⚠️ **Usa tu propia API key de Groq: https://console.groq.com**
   - Guarda los cambios

### 5. Esperar deployment
   - Railway detectará automáticamente Python
   - Instalará dependencias (5-10 minutos)
   - Descargará modelos de ML
   - Iniciará la aplicación

### 6. Obtener URL
   - Ve a "Settings" → "Domains"
   - Railway generará una URL tipo: `usatin35-production.up.railway.app`
   - Copia y accede a tu chatbot

## 📝 NOTAS IMPORTANTES:

⚠️ **Tiempo de build**: 5-10 minutos por los modelos de ML
⚠️ **Primer arranque**: Puede tardar ~30 segundos en cargar modelos
⚠️ **Plan gratuito**: 500 horas/mes, suficiente para demos

## 🔧 Si hay problemas:

### Error de memoria:
Si Railway muestra "Out of Memory":
1. Ve a Settings en Railway
2. Aumenta RAM a 1GB (disponible en plan gratuito)

### Error de timeout:
Si el build tarda mucho:
- Es normal, los modelos son pesados
- Railway permite hasta 10 min de build

### Logs:
Para ver qué está pasando:
- Clic en tu servicio
- Pestaña "Deployments"
- Clic en el deployment activo
- Ver "Build logs" y "Deploy logs"

## 🎉 ¡LISTO!

Una vez deployado, tu chatbot estará disponible 24/7 en la URL de Railway.

Comparte la URL con quien quieras y podrán usar el chatbot desde cualquier lugar.
