# Chatbot RAG USAT - Deployment en Railway

## 🚀 Despliegue en Railway

### Paso 1: Preparar el repositorio en GitHub

1. **Crear un repositorio en GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit - RAG Chatbot USAT"
   git branch -M main
   git remote add origin https://github.com/TU_USUARIO/TU_REPO.git
   git push -u origin main
   ```

### Paso 2: Configurar Railway

1. **Crear cuenta en Railway**
   - Ve a [railway.app](https://railway.app)
   - Regístrate con GitHub (recomendado)

2. **Crear nuevo proyecto**
   - Click en "New Project"
   - Selecciona "Deploy from GitHub repo"
   - Autoriza Railway para acceder a tus repos
   - Selecciona tu repositorio `usatin3.5`

3. **Configurar variables de entorno**
   En Railway, ve a tu proyecto → Variables y agrega:
   ```
   GROQ_API_KEY=tu_api_key_aqui
   PORT=8080
   ```
   
   ⚠️ **IMPORTANTE**: No compartas tu API key públicamente. Usa tu propia key de Groq.

### Paso 3: Deploy automático

Railway detectará automáticamente:
- `requirements.txt` → Instalará dependencias
- `Procfile` → Ejecutará el comando de inicio
- `runtime.txt` → Usará Python 3.9

El deployment tarda aproximadamente 5-10 minutos por los modelos de ML.

### Paso 4: Verificar deployment

1. Railway te dará una URL pública tipo: `https://tu-app.railway.app`
2. Accede y prueba el chatbot

## 📝 Notas importantes

### Vectorstore Cache
- Los archivos en `vectorstore_cache/` están incluidos en el repo
- Railway los mantendrá entre deployments
- Si actualizas documentos, elimina el cache y redeploya

### Documentos
- Los PDFs en `documents/` están incluidos
- Para agregar nuevos documentos:
  1. Agregar archivos a `documents/`
  2. Eliminar `vectorstore_cache/`
  3. Hacer commit y push
  4. Railway redeployará automáticamente

### Límites del plan gratuito
- 500 horas/mes de ejecución
- 512MB RAM
- 1GB storage
- Suficiente para este proyecto

## 🔧 Troubleshooting

### Error: Out of memory
- El modelo de reranking es pesado
- Solución: Comenta la línea `initialize_reranker()` en `app.py`

### Error: Timeout during build
- Los modelos tardan en descargar
- Solución: Railway tiene 10 min de timeout, debería ser suficiente

### Documentos no se cargan
- Verifica que `documents/` tenga los PDFs
- Verifica que `vectorstore_cache/` esté actualizado

## 🔄 Actualizar el deployment

```bash
# Hacer cambios en el código
git add .
git commit -m "Descripción de cambios"
git push

# Railway detectará el push y redeployará automáticamente
```

## 📊 Monitoreo

En Railway puedes ver:
- Logs en tiempo real
- Uso de recursos (CPU, RAM)
- Métricas de requests
- Costos (si sales del plan gratuito)

## 🌐 Custom Domain (Opcional)

1. En Railway → Settings → Domains
2. Agrega tu dominio personalizado
3. Configura DNS según instrucciones

## 🔐 Seguridad

- ✅ La API Key está como variable de entorno (no en código)
- ✅ CORS está configurado
- ⚠️ Considera agregar autenticación si es público
