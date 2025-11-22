#!/usr/bin/env python3
"""
Script para descargar el dataset VQA-Med 2019 desde Google Drive usando gdown.
Descarga el ZIP principal y extrae el Training.zip.
Uso: python scripts/download_vqa_med_2019.py
"""

import os
import sys
import zipfile
from pathlib import Path

def download_dataset():
    """Descarga el dataset VQA-Med 2019 desde Google Drive y extrae Training.zip."""

    print("📥 Descargando dataset VQA-Med 2019 desde Google Drive...")

    # Crear directorio data si no existe
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # ID del archivo en Google Drive
    # Link: https://drive.google.com/file/d/1Xu_Y2Z6lvZGgExxz0VdYY6wFgAFk3oZf/view?usp=sharing
    dataset_id = "1Xu_Y2Z6lvZGgExxz0VdYY6wFgAFk3oZf"
    main_zip = data_dir / "VQA-Med-2019.zip"  # ZIP principal
    training_zip = data_dir / "ImageClef-2019-VQA-Med-Training.zip"  # Training extraído

    print(f"🔗 ID del archivo: {dataset_id}")
    print(f"📁 ZIP principal: {main_zip}")
    print(f"📁 Training extraído: {training_zip}")

    try:
        import gdown

        # Descargar ZIP principal usando gdown
        print("📥 Iniciando descarga del ZIP principal con gdown...")
        gdown.download(
            id=dataset_id,
            output=str(main_zip),
            quiet=False
        )

        # Verificar descarga del ZIP principal
        if not main_zip.exists() or main_zip.stat().st_size < 1000:
            print("❌ Error: El archivo descargado es muy pequeño o está vacío")
            return False

        size_mb = main_zip.stat().st_size / (1024 * 1024)
        print(f"✅ ZIP principal descargado: {size_mb:.1f} MB")
        
        # Extraer Training.zip del ZIP principal
        print("📦 Extrayendo ImageClef-2019-VQA-Med-Training.zip del ZIP principal...")
        training_zip_name = "ImageClef-2019-VQA-Med-Training.zip"
        
        with zipfile.ZipFile(main_zip, "r") as zf:
            # Buscar el archivo Training.zip dentro del ZIP
            if training_zip_name not in zf.namelist():
                # Buscar con diferentes variaciones de nombre
                found = False
                for name in zf.namelist():
                    if "Training" in name and name.endswith(".zip"):
                        training_zip_name = name
                        found = True
                        break
                
                if not found:
                    print(f"❌ Error: No se encontró {training_zip_name} en el ZIP")
                    print(f"   Archivos disponibles en el ZIP:")
                    for name in zf.namelist()[:10]:
                        print(f"     - {name}")
                    return False
            
            # Extraer el Training.zip
            print(f"   Extrayendo: {training_zip_name}")
            with zf.open(training_zip_name) as source:
                with open(training_zip, "wb") as target:
                    target.write(source.read())
        
        # Verificar extracción
        if training_zip.exists() and training_zip.stat().st_size > 1000:
            size_mb = training_zip.stat().st_size / (1024 * 1024)
            print(f"✅ Training.zip extraído exitosamente!")
            print(f"📊 Tamaño del archivo: {size_mb:.1f} MB")
            print("🎉 ¡Listo para usar en los experimentos!")
            
            # Opcional: eliminar el ZIP principal para ahorrar espacio
            # Descomentar si quieres eliminar el ZIP principal después de extraer
            # main_zip.unlink()
            # print("🗑️  ZIP principal eliminado (Training.zip guardado)")
            
            return True
        else:
            print("❌ Error: El Training.zip extraído es muy pequeño o está vacío")
            return False

    except ImportError:
        print("❌ Error: gdown no está instalado")
        print("💡 Instala gdown con: pip install gdown")
        return False
    except Exception as e:
        print(f"❌ Error durante la descarga o extracción: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Función principal."""
    print("=" * 60)
    print("📥 Descargador de Dataset VQA-Med 2019")
    print("=" * 60)
    print()

    success = download_dataset()

    if not success:
        print("\n💡 Alternativas:")
        print("   1. Descargar manualmente desde:")
        print("      https://drive.google.com/file/d/1Xu_Y2Z6lvZGgExxz0VdYY6wFgAFk3oZf/view?usp=sharing")
        print("   2. Usar gdown directamente:")
        print("      gdown 1Xu_Y2Z6lvZGgExxz0VdYY6wFgAFk3oZf -O data/VQA-Med-2019.zip")
        print("   3. Extraer manualmente ImageClef-2019-VQA-Med-Training.zip del ZIP descargado")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())

