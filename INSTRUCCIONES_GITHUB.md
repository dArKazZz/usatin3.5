# 📝 Instrucciones para crear repositorio privado en GitHub

## Pasos para crear el repositorio privado:

### Opción 1: Usar GitHub CLI (Recomendado)

1. **Instalar GitHub CLI:**
```bash
brew install gh
```

2. **Autenticarse:**
```bash
gh auth login
```

3. **Crear y subir el repositorio:**
```bash
cd /Users/darkazzz/Desktop/usatin3.5
gh repo create usatin3.5 --private --source=. --remote=origin --push
```

---

### Opción 2: Manualmente desde GitHub.com

1. **Ve a GitHub.com y crea un nuevo repositorio:**
   - URL: https://github.com/new
   - Nombre: `usatin3.5`
   - Visibilidad: **Private** (Privado)
   - NO inicialices con README, .gitignore o license

2. **Ejecuta estos comandos en la terminal:**

```bash
cd /Users/darkazzz/Desktop/usatin3.5

# Remover el remote anterior
git remote remove origin

# Agregar el nuevo remote (reemplaza TU_USUARIO con tu nombre de usuario de GitHub)
git remote add origin https://github.com/TU_USUARIO/usatin3.5.git

# Verificar la rama actual
git branch

# Si no estás en 'main', renombra la rama
git branch -M main

# Agregar todos los archivos
git add .

# Hacer commit
git commit -m "Initial commit: Usatín 3.5 - Sistema RAG mejorado"

# Subir al repositorio privado
git push -u origin main
```

3. **Si pide autenticación:**
   - Usuario: Tu nombre de usuario de GitHub
   - Contraseña: Tu Personal Access Token (PAT)
   
   Para crear un PAT:
   - Ve a: https://github.com/settings/tokens
   - Generate new token (classic)
   - Selecciona scopes: `repo` (completo)
   - Copia el token y úsalo como contraseña

---

## ✅ Verificación

Después de ejecutar los comandos, verifica que todo esté bien:

```bash
# Ver el remote configurado
git remote -v

# Ver el estado
git status

# Ver el último commit
git log --oneline -1
```

Tu repositorio privado estará en:
`https://github.com/TU_USUARIO/usatin3.5`

---

## 📦 Estado actual del proyecto

✅ Carpeta renombrada a: `usatin3.5`
✅ Código limpio y optimizado
✅ README completo y profesional
✅ Listo para subir a GitHub

---

## 🔐 Importante

Este es un repositorio **PRIVADO**, solo tú podrás verlo.
Si quieres compartirlo con alguien:
1. Ve a Settings → Collaborators
2. Agrega usuarios por su nombre de GitHub
