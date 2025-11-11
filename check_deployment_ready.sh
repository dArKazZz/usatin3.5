#!/bin/bash

echo "🔍 Verificando preparación para deployment en Railway..."
echo ""

# Verificar archivos necesarios
echo "📄 Verificando archivos necesarios:"
files=("requirements.txt" "Procfile" "runtime.txt" "app.py" "documents" "vectorstore_cache")

for file in "${files[@]}"; do
    if [ -e "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file - FALTA"
        exit 1
    fi
done

echo ""

# Verificar que hay documentos
echo "📚 Verificando documentos:"
doc_count=$(ls documents/*.pdf 2>/dev/null | wc -l)
if [ "$doc_count" -gt 0 ]; then
    echo "  ✅ $doc_count documento(s) PDF encontrado(s)"
else
    echo "  ⚠️  No se encontraron documentos PDF"
fi

echo ""

# Verificar vectorstore cache
echo "💾 Verificando vectorstore cache:"
cache_count=$(ls vectorstore_cache/*.pkl 2>/dev/null | wc -l)
if [ "$cache_count" -gt 0 ]; then
    echo "  ✅ Cache encontrado"
else
    echo "  ⚠️  No hay cache. Ejecuta: python process_local_documents.py"
fi

echo ""

# Verificar .gitignore
echo "🔒 Verificando .gitignore:"
if grep -q ".venv" .gitignore && grep -q "__pycache__" .gitignore; then
    echo "  ✅ .gitignore configurado correctamente"
else
    echo "  ⚠️  .gitignore puede necesitar actualización"
fi

echo ""

# Verificar tamaño del proyecto
echo "📦 Verificando tamaño del proyecto:"
project_size=$(du -sh . | cut -f1)
echo "  📊 Tamaño total: $project_size"
echo "  ℹ️  Railway soporta hasta 1GB en plan gratuito"

echo ""

# Verificar Git
echo "🔧 Verificando Git:"
if [ -d ".git" ]; then
    echo "  ✅ Git inicializado"
    
    # Verificar si hay cambios sin commit
    if git diff-index --quiet HEAD -- 2>/dev/null; then
        echo "  ✅ No hay cambios sin commit"
    else
        echo "  ⚠️  Hay cambios sin commit"
        echo "     Ejecuta: git add . && git commit -m 'Ready for deployment'"
    fi
    
    # Verificar remote
    if git remote get-url origin > /dev/null 2>&1; then
        remote_url=$(git remote get-url origin)
        echo "  ✅ Remote configurado: $remote_url"
    else
        echo "  ⚠️  No hay remote configurado"
        echo "     Ejecuta: git remote add origin https://github.com/TU_USUARIO/TU_REPO.git"
    fi
else
    echo "  ❌ Git no inicializado"
    echo "     Ejecuta: git init"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📋 SIGUIENTE PASO:"
echo ""
echo "1. Si aún no tienes Git configurado:"
echo "   git init"
echo "   git add ."
echo "   git commit -m 'Initial commit - RAG Chatbot USAT'"
echo ""
echo "2. Crear repositorio en GitHub y pushearlo:"
echo "   git remote add origin https://github.com/TU_USUARIO/TU_REPO.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "3. Ir a https://railway.app y:"
echo "   - Crear cuenta / Login con GitHub"
echo "   - New Project → Deploy from GitHub"
echo "   - Seleccionar tu repositorio"
echo "   - Agregar variable de entorno: GROQ_API_KEY"
echo ""
echo "4. ¡Esperar el deployment y disfrutar! 🎉"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
