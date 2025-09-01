# RClip + SHAP: Clasificación Médica con Balance Multimodal
# 
# Este script demuestra:
# - Clasificación de imágenes médicas usando RClip
# - Análisis de explicabilidad con SHAP
# - Medición del balance multimodal (TScore/IScore)
# - Visualización de mapas de calor para parches de imagen y tokens de texto
#
# Dataset: ROCO (Radiology Objects in COntext)  
# Modelo: RClip (kaveh/rclip)
#
# 🎯 Configuración CPU-First para estabilidad
# Este script está configurado para usar CPU forzado para evitar errores de CUDA

# =============================================================================
# CONFIGURACIÓN INICIAL
# =============================================================================

print("🚀 Iniciando configuración...")

# Montar Google Drive
from google.colab import drive
drive.mount('/content/drive')

# =============================================================================
# CLONAR REPOSITORIO
# =============================================================================

print("📥 Clonando repositorio...")

REPO_URL = "https://github.com/Alberto-97sc/mmshap_medclip.git"
LOCAL_DIR = "/content/mmshap_medclip"
BRANCH = "others-clips-version"

%cd /content
import os, shutil, subprocess, sys

if not os.path.isdir(f"{LOCAL_DIR}/.git"):
    !git clone $REPO_URL $LOCAL_DIR
else:
    %cd $LOCAL_DIR
    !git fetch origin
    !git checkout $BRANCH
    !git reset --hard origin/$BRANCH

%cd $LOCAL_DIR
commit_hash = !git rev-parse --short HEAD
print(f"✅ Repositorio actualizado: commit {commit_hash[0]}")

# =============================================================================
# INSTALAR DEPENDENCIAS
# =============================================================================

print("📦 Instalando dependencias...")

%pip install -e /content/mmshap_medclip
%pip install tqdm

print("✅ Dependencias instaladas")

# =============================================================================
# CARGAR DATOS Y MODELO
# =============================================================================

print("🤖 Cargando modelo y datos...")

CFG_PATH = "/content/mmshap_medclip/configs/roco_classification_rclip.yaml"

from mmshap_medclip.io_utils import load_config
from mmshap_medclip.devices import get_device
from mmshap_medclip.registry import build_dataset, build_model

# Cargar configuración
cfg = load_config(CFG_PATH)
device = get_device()
print(f"🖥️ Dispositivo: {device}")

# Cargar dataset
print("📁 Cargando dataset ROCO...")
dataset = build_dataset(cfg["dataset"])
print(f"✅ Dataset cargado: {len(dataset)} muestras")

# Cargar modelo
print("🤖 Cargando modelo RClip...")
model = build_model(cfg["model"], device=device)
print("✅ Modelo RClip cargado")

# =============================================================================
# DEFINIR CLASES
# =============================================================================

print("🏷️ Configurando clases...")

class_names = [
    "Chest X-Ray", "Brain MRI", "Abdominal CT Scan",
    "Ultrasound", "OPG", "Mammography", "Bone X-Ray"
]

print(f"🏷️ Clases definidas: {len(class_names)}")
for i, clase in enumerate(class_names):
    print(f"  {i+1}. {clase}")

# =============================================================================
# CARGAR MUESTRA Y MOSTRAR IMAGEN
# =============================================================================

print("🖼️ Cargando muestra...")

from mmshap_medclip.tasks.classification import run_classification_one
import matplotlib.pyplot as plt

muestra_idx = 266
sample = dataset[muestra_idx]
image = sample['image']
caption = sample['text']

print(f"📋 Muestra {muestra_idx}:")
print(f"Caption original: {caption[:100]}...")

# Mostrar imagen
plt.figure(figsize=(8, 6))
plt.imshow(image)
plt.title(f"Muestra {muestra_idx} - ROCO Dataset")
plt.axis('off')
plt.show()

# =============================================================================
# EJECUTAR CLASIFICACIÓN CON SHAP
# =============================================================================

print("🔬 Ejecutando clasificación con SHAP...")

# Cargar configuración SHAP desde el YAML
from mmshap_medclip.tasks.classification import load_shap_config_from_yaml
shap_config = load_shap_config_from_yaml(CFG_PATH)

print(f"⚙️ Configuración SHAP cargada:")
print(f"  - Algorithm: {shap_config.get('algorithm', 'permutation')}")
print(f"  - Max evals: {shap_config.get('max_evals', 1000)}")
print(f"  - Force CPU: {shap_config.get('force_cpu', False)}")
print(f"  - Hybrid mode: {shap_config.get('hybrid_mode', False)}")

# Ejecutar clasificación
res_shap = run_classification_one(
    model, image, class_names, device, 
    explain=True, plot=True, shap_config=shap_config
)

# =============================================================================
# MOSTRAR RESULTADOS
# =============================================================================

print(f"\n🎯 Resultados:")
print(f"Clase predicha: {res_shap['predicted_class']}")
print(f"Confianza: {res_shap['probabilities'].max():.2%}")
print(f"TScore (Texto): {res_shap['tscore']:.2%}")
print(f"IScore (Imagen): {res_shap['iscore']:.2%}")

# Interpretación del balance
if res_shap['tscore'] > 0.6:
    balance_msg = "🔤 Enfoque en TEXTO"
elif res_shap['iscore'] > 0.6:
    balance_msg = "🖼️ Enfoque en IMAGEN"
else:
    balance_msg = "⚖️ Balance equilibrado"
    
print(f"Balance: {balance_msg}")

# Mostrar probabilidades de todas las clases
print(f"\n📊 Probabilidades por clase:")
for clase, prob in zip(class_names, res_shap['probabilities']):
    bar = "█" * int(prob * 20)
    print(f"  {clase:<20}: {prob:.2%} {bar}")

# Mostrar mapa de calor
if 'fig' in res_shap:
    print("\n🖼️ Mostrando mapa de calor...")
    display(res_shap['fig'])

# =============================================================================
# RESUMEN FINAL
# =============================================================================

print("\n" + "="*60)
print("🎉 ¡CLASIFICACIÓN COMPLETADA EXITOSAMENTE!")
print("="*60)
print(f"✅ Modelo: RClip")
print(f"✅ Dataset: ROCO ({len(dataset)} muestras)")
print(f"✅ Clasificación: {res_shap['predicted_class']}")
print(f"✅ TScore: {res_shap['tscore']:.2%}")
print(f"✅ IScore: {res_shap['iscore']:.2%}")
print(f"✅ Balance: {balance_msg}")
print("="*60)
print("🚀 Próximo paso: Cambiar force_cpu: false en YAML para GPU")
print("="*60)
