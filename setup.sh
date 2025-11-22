#!/bin/bash

# =============================================================================
# Script de inicialización completa para mmshap_medclip
# =============================================================================
# Este script configura todo el entorno necesario:
# - Instalación de Python (si no está presente)
# - Configuración de Git
# - Instalación de dependencias directamente en el sistema
# - Descarga del dataset
# - Conversión de scripts a notebooks
# =============================================================================

set -e  # Salir si algún comando falla

# Obtener el directorio del script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# =============================================================================
# Funciones auxiliares para mostrar progreso
# =============================================================================

# Función para mostrar barra de progreso
show_progress() {
    local current=$1
    local total=$2
    local width=50
    local percentage=$((current * 100 / total))
    local completed=$((width * current / total))
    local remaining=$((width - completed))

    printf "\r   Progreso: ["
    printf "%${completed}s" | tr ' ' '█'
    printf "%${remaining}s" | tr ' ' '░'
    printf "] %3d%%" "$percentage"
}

# Función para mostrar spinner durante operaciones largas
show_spinner() {
    local pid=$1
    local message=$2
    local spin='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    local i=0

    while kill -0 $pid 2>/dev/null; do
        i=$(((i + 1) % 10))
        printf "\r   ${spin:$i:1} $message..."
        sleep 0.1
    done
    printf "\r"
}

# Contadores de progreso general
TOTAL_STEPS=5
CURRENT_STEP=0
START_TIME=$(date +%s)

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   Inicializando proyecto mmshap_medclip                        ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "   ⏱️  Tiempo estimado total: 15-25 minutos"
echo "   💡 El script mostrará el progreso de cada paso"
echo ""
show_progress 0 $TOTAL_STEPS
echo ""
echo ""

# =============================================================================
# 1. Verificar e instalar Python si es necesario
# =============================================================================
echo "🐍 [1/5] Verificando instalación de Python..."

if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo "   ✅ Python ya está instalado (versión $PYTHON_VERSION)"
else
    echo "   ⚠️  Python no encontrado. Instalando Python3..."

    # Detectar sistema operativo
    if [ -f /etc/debian_version ]; then
        # Debian/Ubuntu
        sudo apt-get update -qq
        sudo apt-get install -y python3 python3-pip python3-venv
        echo "   ✅ Python3 instalado correctamente"
    elif [ -f /etc/redhat-release ]; then
        # RedHat/CentOS/Fedora
        sudo yum install -y python3 python3-pip
        echo "   ✅ Python3 instalado correctamente"
    else
        echo "   ❌ No se pudo detectar el gestor de paquetes"
        echo "      Por favor, instala Python3 manualmente y vuelve a ejecutar este script"
        exit 1
    fi

    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo "      Versión instalada: $PYTHON_VERSION"
fi

# Verificar y instalar python3-venv si no está disponible
if ! python3 -m venv --help &> /dev/null; then
    echo "   → Instalando python3-venv..."
    if [ -f /etc/debian_version ]; then
        sudo apt-get update -qq
        sudo apt-get install -y python3-venv
    elif [ -f /etc/redhat-release ]; then
        sudo yum install -y python3-venv
    fi
    echo "   ✅ python3-venv instalado"
fi
echo ""

# Actualizar progreso
CURRENT_STEP=1
show_progress $CURRENT_STEP $TOTAL_STEPS
echo ""
echo ""

# =============================================================================
# 2. Configuración de Git
# =============================================================================
echo "📝 [2/5] Configurando Git..."
git config user.name "Alberto-97sc"
git config user.email "alberthg.ramos@gmail.com"
echo "   ✅ Git configurado correctamente"
echo "      Usuario: $(git config user.name)"
echo "      Email: $(git config user.email)"
echo ""

# Actualizar progreso
CURRENT_STEP=2
show_progress $CURRENT_STEP $TOTAL_STEPS
echo ""
echo ""

# =============================================================================
# 3. Instalar dependencias
# =============================================================================
echo "📦 [3/5] Instalando dependencias..."
echo "   ⏱️  Esta operación puede tardar 5-10 minutos..."

# Actualizar pip
echo "   → Actualizando pip..."
python3 -m pip install --upgrade pip --quiet 2>&1 | grep -v "^Requirement" || true
echo "      ✓ pip actualizado"

# Instalar proyecto con soporte para notebooks
echo ""
echo "   → Instalando dependencias principales..."
echo "      • torch, torchvision (modelos deep learning)"
echo "      • transformers (modelos CLIP)"
echo "      • SHAP (explicabilidad)"
echo "      • jupytext, notebook (notebooks interactivos)"
echo ""
echo "   ⏳ Descargando e instalando paquetes..."
echo "      (Esto tardará varios minutos, por favor espera)"
echo ""

# Ejecutar pip install y mostrar progreso simple
echo "      → Instalando paquete principal en modo editable..."
pip install -e . 2>&1 | while IFS= read -r line; do
    if echo "$line" | grep -q "Collecting\|Downloading\|Installing collected\|Successfully installed mmshap_medclip"; then
        echo "      → $line"
    fi
done

echo ""
echo "      → Instalando dependencias adicionales para notebooks..."
pip install jupytext notebook ipykernel 2>&1 | while IFS= read -r line; do
    if echo "$line" | grep -q "Collecting\|Downloading\|Installing collected"; then
        echo "      → $line"
    fi
done

echo ""
echo "   ✅ Dependencias instaladas correctamente"
echo "      ✓ Paquete mmshap_medclip en modo editable"
echo "      ✓ Dependencias para notebooks (jupytext, jupyter)"
echo ""

# Actualizar progreso
CURRENT_STEP=3
show_progress $CURRENT_STEP $TOTAL_STEPS
echo ""
echo ""

# =============================================================================
# 5. Descargar datasets
# =============================================================================
echo "📥 [4/5] Descargando datasets desde Google Drive..."
echo ""

# =============================================================================
# 5.1. Descargar dataset ROCO
# =============================================================================
echo "   📥 [4.1/5] Descargando dataset ROCO..."
echo "      ⏱️  Tamaño del dataset: ~6.65 GB (puede tardar 5-15 minutos)"
echo ""

if [ -f "data/dataset_roco.zip" ]; then
    echo "      ℹ️  El dataset ROCO ya existe en data/dataset_roco.zip"
    read -p "      ¿Deseas volver a descargarlo? (s/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Ss]$ ]]; then
        echo "      → Descargando dataset ROCO..."
        echo "         (gdown mostrará su propio progreso de descarga)"
        echo ""
        python3 scripts/download_dataset.py
        echo ""
        echo "      ✅ Dataset ROCO descargado nuevamente"
    else
        echo "      ↩️  Se usará el dataset ROCO existente"
    fi
else
    echo "      → Descargando dataset ROCO..."
    echo "         (gdown mostrará su propio progreso de descarga)"
    echo ""
    python3 scripts/download_dataset.py
    echo ""
    echo "      ✅ Dataset ROCO descargado correctamente"
fi
echo ""

# =============================================================================
# 5.2. Descargar dataset MedVQA 2019
# =============================================================================
echo "   📥 [4.2/5] Descargando dataset MedVQA 2019..."
echo "      ⏱️  Tamaño del dataset: variable (puede tardar varios minutos)"
echo ""

if [ -f "data/VQA-Med-2019.zip" ]; then
    echo "      ℹ️  El dataset MedVQA 2019 ya existe en data/VQA-Med-2019.zip"
    read -p "      ¿Deseas volver a descargarlo? (s/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Ss]$ ]]; then
        echo "      → Descargando dataset MedVQA 2019..."
        echo "         (gdown mostrará su propio progreso de descarga)"
        echo ""
        python3 scripts/download_vqa_med_2019.py
        echo ""
        echo "      ✅ Dataset MedVQA 2019 descargado nuevamente"
    else
        echo "      ↩️  Se usará el dataset MedVQA 2019 existente"
    fi
else
    echo "      → Descargando dataset MedVQA 2019..."
    echo "         (gdown mostrará su propio progreso de descarga)"
    echo ""
    python3 scripts/download_vqa_med_2019.py
    echo ""
    echo "      ✅ Dataset MedVQA 2019 descargado correctamente"
fi
echo ""

# Actualizar progreso
CURRENT_STEP=4
show_progress $CURRENT_STEP $TOTAL_STEPS
echo ""
echo ""

# =============================================================================
# 5. Convertir scripts a notebooks
# =============================================================================
echo "📓 [5/5] Convirtiendo scripts a notebooks Jupyter..."

if command -v jupytext &> /dev/null; then
    echo "   → Convirtiendo archivos .py a .ipynb..."
    jupytext --to notebook experiments/*.py 2>/dev/null || true
    echo "   ✅ Notebooks creados en experiments/"
    echo "      - experiments/pubmedclip_roco_isa.ipynb"
    echo "      - experiments/whyxrayclip_roco_isa.ipynb"
else
    echo "   ⚠️  jupytext no está disponible"
    echo "      Ejecuta manualmente:"
    echo "      jupytext --to notebook experiments/*.py"
fi
echo ""

# Actualizar progreso final
CURRENT_STEP=5
show_progress $CURRENT_STEP $TOTAL_STEPS
echo ""
echo ""

# =============================================================================
# Finalización
# =============================================================================

# Calcular tiempo transcurrido
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   ✅ INSTALACIÓN COMPLETADA EXITOSAMENTE                       ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
printf "   ⏱️  Tiempo transcurrido: %d minutos y %d segundos\n" $MINUTES $SECONDS
echo ""
echo "🎯 Próximos pasos:"
echo ""
echo "   1. Ejecutar un experimento:"
echo "      $ python3 experiments/pubmedclip_roco_isa.py"
echo "      $ python3 experiments/whyxrayclip_roco_isa.py"
echo ""
echo "   2. O usar Jupyter Notebook:"
echo "      $ jupyter notebook"
echo "      Luego abrir: experiments/pubmedclip_roco_isa.ipynb"
echo "      Seleccionar cualquier kernel de Python 3.12"
echo ""
echo "📚 Para más información, consulta el README.md"
echo ""
