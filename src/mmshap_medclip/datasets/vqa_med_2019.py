import os
import zipfile
from PIL import Image
from io import BytesIO
from typing import Dict, List
from collections import defaultdict
from mmshap_medclip.datasets.base import DatasetBase
from mmshap_medclip.registry import register_dataset

@register_dataset("vqa_med_2019")
def _build_vqa_med_2019(params):
    return VQAMed2019Dataset(**params)

class VQAMed2019Dataset(DatasetBase):
    """
    Dataset loader para VQA-Med 2019.
    
    SOLO soporta el split TRAINING.
    
    Lee archivos del ZIP ImageClef-2019-VQA-Med-Training.zip:
    - Archivos QAPairsByCategory/*_train.txt (C1_Modality_train.txt, C2_Plane_train.txt, C3_Organ_train.txt)
    - Directorio Train_images/
    
    También puede leer desde el ZIP padre (VQA-Med-2019.zip) que contiene los zips hijos.
    
    Infiere categorías desde el nombre del archivo y construye candidatos por categoría.
    """
    
    VALID_CATEGORIES = {"modality", "plane", "organ_system"}

    def __init__(
        self,
        zip_path: str,
        split: str = "Training",
        images_subdir: str = None,
        n_rows: str = "all"
    ):
        """
        Args:
            zip_path: Ruta al archivo ZIP del dataset (puede ser el zip padre VQA-Med-2019.zip o el zip hijo)
            split: Split a usar (SOLO se soporta 'Training' o 'train')
            images_subdir: Subdirectorio dentro del ZIP donde están las imágenes
                          Si es None, se usa "Train_images" (único soportado)
            n_rows: Número de filas a cargar ("all" o un entero)
        """
        # FORZAR split a Training - solo se soporta training
        split_lower = split.lower()
        if split_lower not in ["training", "train"]:
            raise ValueError(
                f"El split '{split}' no está soportado. "
                f"Este dataset solo soporta el split TRAINING. "
                f"Especifica split='Training' o split='train'."
            )
        
        # Normalizar a "Training"
        self.split = "Training"
        self.zip_path = zip_path
        
        # Inicializar candidates_per_cat como dict vacío
        self.candidates_per_cat = {}
        
        # Inferir images_subdir si no se proporciona
        # SOLO se soporta Train_images (split Training)
        if images_subdir is None:
            self.images_subdir = "Train_images"
        else:
            self.images_subdir = images_subdir
            # Verificar que no se esté intentando usar Val_images u otro directorio
            if "val" in images_subdir.lower() or "validation" in images_subdir.lower():
                raise ValueError(
                    f"El subdirectorio '{images_subdir}' no está soportado. "
                    f"Este dataset solo soporta 'Train_images' para el split Training."
                )
        
        # Detectar si el zip_path es el zip padre (VQA-Med-2019.zip) que contiene zips hijos
        # En ese caso, necesitamos abrir el zip hijo correspondiente
        self.is_nested_zip = False
        self.inner_zip_name = None
        self.inner_zip_data = None  # Guardar el zip hijo en memoria para uso posterior
        
        # Verificar si es el zip padre
        zip_basename = os.path.basename(zip_path).lower()
        # Detectar si es el zip padre (VQA-Med-2019.zip)
        # También verificar si el archivo existe y si contiene zips hijos
        is_vqa_med_2019_zip = "vqa-med-2019" in zip_basename and zip_basename.endswith(".zip")
        
        if is_vqa_med_2019_zip:
            # Verificar que el archivo existe y contiene zips hijos
            if os.path.exists(zip_path):
                try:
                    with zipfile.ZipFile(zip_path, "r") as test_zip:
                        has_nested_zips = any(name.endswith(".zip") for name in test_zip.namelist())
                        if has_nested_zips:
                            self.is_nested_zip = True
                except:
                    # Si no se puede abrir, asumir que no es anidado
                    self.is_nested_zip = False
            else:
                self.is_nested_zip = False
        else:
            self.is_nested_zip = False
        
        if self.is_nested_zip:
            # SOLO se soporta Training
            self.inner_zip_name = "ImageClef-2019-VQA-Med-Training.zip"
        
        # Detectar el prefijo de directorio raíz del ZIP si existe
        # Por ejemplo: "ImageClef-2019-VQA-Med-Training/"
        self.zip_root_prefix = None
        
        # SOLO se soporta Training - forzar split a "train"
        self.detected_split = "train"
        print(f"📊 Split: TRAINING (usará solo archivos *train.txt)")
        
        # Cargar preguntas y respuestas desde el ZIP
        # Si es un zip anidado, abrir el zip padre y luego el zip hijo
        if self.is_nested_zip:
            # Abrir el zip padre
            with zipfile.ZipFile(zip_path, "r") as parent_zip:
                # Verificar que el zip hijo Training existe
                if self.inner_zip_name not in parent_zip.namelist():
                    # Buscar solo Training (no validation ni test)
                    found = False
                    for name in parent_zip.namelist():
                        if "training" in name.lower() and name.endswith(".zip"):
                            self.inner_zip_name = name
                            found = True
                            break
                    if not found:
                        raise FileNotFoundError(
                            f"No se encontró el zip hijo ImageClef-2019-VQA-Med-Training.zip en {zip_path}. "
                            f"Archivos disponibles: {parent_zip.namelist()[:10]}"
                        )
                
                # Leer el zip hijo en memoria y guardarlo para uso posterior
                self.inner_zip_data = parent_zip.read(self.inner_zip_name)
                # Abrir el zip hijo desde memoria
                zf = zipfile.ZipFile(BytesIO(self.inner_zip_data), "r")
        else:
            # Abrir directamente el zip hijo
            zf = zipfile.ZipFile(zip_path, "r")
        
        try:
            # Buscar archivos QA por categoría dentro de QAPairsByCategory
            all_txt_files = [n for n in zf.namelist() if n.endswith(".txt")]
            all_files = zf.namelist()  # Todos los archivos para debugging
            
            # Detectar prefijo de directorio raíz del ZIP (ej: "ImageClef-2019-VQA-Med-Training/")
            if all_files:
                first_file = all_files[0]
                if '/' in first_file:
                    parts = first_file.split('/')
                    if len(parts) > 1:
                        self.zip_root_prefix = parts[0] + '/'
                        print(f"📂 Detectado prefijo de directorio en ZIP: {self.zip_root_prefix}")
            
            category_files = []
            for name in all_txt_files:
                dirname = os.path.dirname(name).lower()
                basename = os.path.basename(name)
                basename_lower = basename.lower()
                
                # Asegurar que provenga de QAPairsByCategory y que sea de categorías C1-C3
                if "qapairsbycategory" not in dirname:
                    continue
                if not basename.startswith("C") or len(basename) < 2:
                    continue
                if "c4" in basename_lower or "abnormality" in basename_lower:
                    continue  # ignorar abnormality
                
                # Filtrar estrictamente por split train
                if not basename_lower.endswith(f"{self.detected_split}.txt"):
                    continue
                if f"_{self.detected_split}.txt" not in basename_lower and f"-{self.detected_split}.txt" not in basename_lower:
                    continue
                
                category_files.append(name)
            
            if not category_files:
                txt_files = [n for n in zf.namelist() if n.endswith(".txt")]
                dirs = sorted(set([os.path.dirname(n) for n in zf.namelist() if os.path.dirname(n)]))
                
                error_msg = (
                    "No se encontraron archivos QAPairsByCategory para el split TRAINING.\n"
                    "Se requieren archivos C1/C2/C3 *_train.txt dentro de QAPairsByCategory/.\n"
                    f"Archivos .txt disponibles ({len(txt_files)}):\n" +
                    "\n".join(f"  - {f}" for f in txt_files[:20]) +
                    (f"\n  ... y {len(txt_files) - 20} más" if len(txt_files) > 20 else "") +
                    f"\n\nDirectorios en el ZIP ({len(dirs)}):\n" +
                    "\n".join(f"  - {d}" for d in dirs[:10]) +
                    (f"\n  ... y {len(dirs) - 10} más" if len(dirs) > 10 else "")
                )
                raise FileNotFoundError(error_msg)
            
            category_files = sorted(category_files)
            print(f"📁 Usando archivos por categoría (TRAINING): {[os.path.basename(f) for f in category_files]}")
            
            files_to_read = []
            for f in category_files:
                basename_lower = os.path.basename(f).lower()
                if basename_lower.endswith(f"{self.detected_split}.txt"):
                    files_to_read.append(f)
            
            if not files_to_read:
                raise FileNotFoundError(
                    "No se pudo construir la lista de archivos *_train.txt para QAPairsByCategory.\n"
                    f"Archivos detectados: {[os.path.basename(f) for f in category_files]}"
                )
            
            print(f"📁 Archivos a leer para split TRAINING: {len(files_to_read)} archivos")
            for f in files_to_read:
                print(f"   - {os.path.basename(f)}")
            
            # Formato esperado: image_id|question|answer
            self.samples = []
            
            for file_to_read in files_to_read:
                # Inferir categoría desde el nombre del archivo
                basename = os.path.basename(file_to_read).lower()
                category = None
                
                if "c1" in basename or "modality" in basename:
                    category = "modality"
                elif "c2" in basename or "plane" in basename:
                    category = "plane"
                elif "c3" in basename or "organ" in basename:
                    category = "organ_system"
                elif "c4" in basename or "abnormality" in basename:
                    # IGNORAR archivos de abnormality completamente
                    continue
                
                if category is None:
                    # Si no se puede inferir desde el nombre, saltar este archivo
                    print(f"⚠️  Advertencia: No se pudo inferir categoría desde {file_to_read}, saltando...")
                    continue
                
                with zf.open(file_to_read) as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.decode('utf-8').strip()
                        if not line:
                            continue
                        
                        # Parsear formato: image_id|question|answer
                        try:
                            parts = line.split("|")
                            if len(parts) != 3:
                                if line_num <= 5:
                                    print(f"⚠️  Advertencia: Línea {line_num} no tiene formato image_id|question|answer: {line[:80]}")
                                continue
                            
                            image_id = parts[0].strip()
                            question = parts[1].strip()
                            answer = parts[2].strip()
                            
                            # Validar que tenemos los campos mínimos
                            if not image_id or not question or not answer:
                                if line_num <= 5:
                                    print(f"⚠️  Advertencia: Campos vacíos en línea {line_num}: {line[:80]}")
                                continue
                            
                            # La categoría ya viene normalizada desde el nombre del archivo
                            # Asegurar que siempre sea una de las categorías válidas
                            if category not in ["modality", "plane", "organ_system"]:
                                # Esto no debería ocurrir, pero por seguridad
                                print(f"⚠️  Advertencia: Categoría inesperada '{category}' en archivo {file_to_read}, saltando muestra")
                                continue
                            
                            self.samples.append({
                                'question_id': image_id,  # Usar image_id como question_id
                                'question': question,
                                'answer': answer,
                                'category': category,  # Categoría normalizada desde nombre de archivo
                                'image_filename': image_id  # image_id es el nombre de la imagen
                            })
                        except Exception as e:
                            if line_num <= 5:
                                print(f"⚠️  Error parseando línea {line_num}: {e} - {line[:80]}")
                            continue
            
            # Construir índice de imágenes (basename -> ruta completa)
            # Buscar en cualquier ubicación, pero priorizar el subdirectorio correcto
            self._name_to_path = {}
            for name in zf.namelist():
                if name.endswith("/") or not name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                base = os.path.basename(name)
                # Priorizar imágenes en el subdirectorio correcto (puede estar en cualquier nivel)
                # Buscar "Train_images" o "images_subdir" en cualquier parte de la ruta
                score = int(self.images_subdir.lower() in name.lower())
                # Bonus si está en el directorio raíz detectado
                if self.zip_root_prefix and name.startswith(self.zip_root_prefix):
                    score += 1
                prev = self._name_to_path.get(base)
                if prev is None or score > prev[0]:
                    self._name_to_path[base] = (score, name)
            self._name_to_path = {k: v[1] for k, v in self._name_to_path.items()}
            
            # Construir candidatos por categoría DESPUÉS de aplicar el filtrado por split y categoría
            # Esto inicializa self.candidates_per_cat
            print(f"📊 Construyendo candidatos desde {len(self.samples)} muestras del split TRAINING...")
            self._build_candidates_by_category()
            self._filter_samples_without_candidates()
            self._build_candidates_by_category()
            
            # Limitar número de muestras si se especifica
            if n_rows != "all":
                self.samples = self.samples[:int(n_rows)]
                # Reconstruir candidatos después de limitar muestras
                print(f"📊 Reconstruyendo candidatos después de limitar a {n_rows} muestras...")
                self._build_candidates_by_category()
                self._filter_samples_without_candidates()
                self._build_candidates_by_category()
        finally:
            # Cerrar el zip si fue abierto
            if zf:
                zf.close()
    
    def _infer_category_from_filename(self, filename: str) -> str:
        """
        Infiere la categoría desde el nombre del archivo.
        
        Categorías según VQA-Med 2019:
        - C1_Modality_* → "modality"
        - C2_Plane_* → "plane"
        - C3_Organ_* → "organ_system"
        - C4_Abnormality_* → "abnormality"
        """
        basename = os.path.basename(filename).lower()
        
        if "c1" in basename or "modality" in basename:
            return "modality"
        elif "c2" in basename or "plane" in basename:
            return "plane"
        elif "c3" in basename or "organ" in basename:
            return "organ_system"
        elif "c4" in basename or "abnormality" in basename:
            return "abnormality"
        
        # No usar "other", lanzar error si no se puede inferir
        raise ValueError(f"No se pudo inferir categoría desde el nombre de archivo: {filename}")
    
    def _build_candidates_by_category(self) -> Dict[str, List[str]]:
        """
        Construye la lista de candidatos válidos por categoría.
        Todas las respuestas únicas de esa categoría dentro del split.
        IGNORA la categoría "abnormality" completamente.
        """
        candidates_by_category = defaultdict(set)
        
        for sample in self.samples:
            category = sample.get('category')
            answer = sample.get('answer')
            # IGNORAR muestras de abnormality
            if category == "abnormality":
                continue
            if category and answer:
                # Agregar respuesta a los candidatos de su categoría
                candidates_by_category[category].add(answer)
        
        # Convertir sets a listas ordenadas
        self.candidates_per_cat = {
            category: sorted(list(answers))
            for category, answers in candidates_by_category.items()
        }
        
        # Debug: mostrar estadísticas y resumen solicitado
        print(f"📊 Construyendo candidatos desde {len(self.samples)} muestras...")
        if self.candidates_per_cat:
            print("Resumen de candidatos por categoría:")
            for cat, cands in self.candidates_per_cat.items():
                print(f"  - {cat}: {len(cands)} candidatos")
                if len(cands) <= 10:
                    print(f"      Ejemplos: {cands[:5]}")
        else:
            print(f"⚠️  ADVERTENCIA: No se construyeron candidatos. Muestras: {len(self.samples)}")
            if self.samples:
                print(f"   Primera muestra: {self.samples[0]}")
        
        # Verificar que las claves de candidates_per_cat coinciden con las categorías en samples
        categories_in_samples = set(s.get('category') for s in self.samples if s.get('category') != "abnormality")
        categories_in_candidates = set(self.candidates_per_cat.keys())
        
        print(f"📊 Verificación de categorías:")
        print(f"   Categorías en samples: {sorted(categories_in_samples)}")
        print(f"   Categorías en candidates_per_cat: {sorted(categories_in_candidates)}")
        
        if categories_in_samples != categories_in_candidates:
            missing_in_candidates = categories_in_samples - categories_in_candidates
            missing_in_samples = categories_in_candidates - categories_in_samples
            if missing_in_candidates:
                print(f"⚠️  ADVERTENCIA: Categorías en samples pero no en candidates_per_cat: {sorted(missing_in_candidates)}")
            if missing_in_samples:
                print(f"⚠️  ADVERTENCIA: Categorías en candidates_per_cat pero no en samples: {sorted(missing_in_samples)}")
        
        return self.candidates_per_cat

    def _filter_samples_without_candidates(self) -> None:
        """
        Elimina cualquier muestra cuya categoría no tenga candidatos disponibles.
        """
        filtered_samples = []
        for sample in self.samples:
            cat = sample.get("category")
            if cat not in self.candidates_per_cat or len(self.candidates_per_cat[cat]) == 0:
                answer = sample.get("answer")
                print(f"⚠️ Eliminando muestra sin candidatos: cat={cat}, answer={answer}")
                continue
            if cat not in self.VALID_CATEGORIES:
                print(f"⚠️ Eliminando muestra con categoría inválida: cat={cat}")
                continue
            filtered_samples.append(sample)
        self.samples = filtered_samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        question = sample['question']
        answer = sample['answer']
        category = sample.get('category')
        if not category:
            raise ValueError(f"Muestra {idx} carece de categoría válida.")
        category = category.strip()
        if category not in self.VALID_CATEGORIES:
            raise ValueError(
                f"Muestra {idx} tiene categoría '{category}' fuera del conjunto permitido {sorted(self.VALID_CATEGORIES)}."
            )
        question_id = sample.get('question_id')
        if not question_id:
            raise ValueError(f"Muestra {idx} carece de question_id.")
        image_filename = sample.get('image_filename')
        
        # Asegurar que candidates_per_cat esté inicializado
        if not hasattr(self, 'candidates_per_cat') or self.candidates_per_cat is None:
            self._build_candidates_by_category()
        
        # ANTES de devolver la muestra: verificar que category existe en candidates_per_cat
        if category not in self.candidates_per_cat:
            print(f"⚠️  ADVERTENCIA: Muestra {idx} tiene categoría '{category}' que no existe en candidates_per_cat")
            print(f"   - image_id: {question_id}")
            print(f"   - question: {question[:80]}...")
            print(f"   - answer: {answer}")
            print(f"   Categorías disponibles: {sorted(self.candidates_per_cat.keys())}")
            raise ValueError(
                f"Muestra {idx} (image_id={question_id}) tiene categoría '{category}' que no existe en candidates_per_cat. "
                f"Esta muestra debería haber sido filtrada durante la construcción del dataset. "
                f"Categorías disponibles: {sorted(self.candidates_per_cat.keys())}"
            )
        
        candidates = self.candidates_per_cat.get(category)
        if not candidates:
            raise ValueError(
                f"Dataset inconsistente: categoría {category} no tiene candidatos."
            )
        # entregar copia para evitar mutaciones externas
        candidates = list(candidates)
        
        # Intentar encontrar la imagen asociada
        image_path = None
        image_filename = sample.get('image_filename')
        
        # Abrir el zip correcto (padre o hijo)
        if self.is_nested_zip:
            # Abrir el zip hijo desde memoria
            zf = zipfile.ZipFile(BytesIO(self.inner_zip_data), "r")
        else:
            # Abrir directamente el zip hijo
            zf = zipfile.ZipFile(self.zip_path, "r")
        
        try:
            # Estrategia 1: Si tenemos el nombre de imagen del archivo de preguntas
            if image_filename:
                # Buscar en el subdirectorio de imágenes
                if self.images_subdir:
                    candidate = f"{self.images_subdir.rstrip('/')}/{image_filename}"
                    if candidate in zf.namelist():
                        image_path = candidate
                
                # Si no se encontró, buscar por basename en el índice
                if image_path is None:
                    base = os.path.basename(image_filename)
                    image_path = self._name_to_path.get(base)
                
                # Si aún no se encontró, buscar por nombre completo
                if image_path is None:
                    image_candidate_paths = [
                        n for n in zf.namelist()
                        if n.endswith(image_filename) or os.path.basename(n) == image_filename
                    ]
                    if image_candidate_paths:
                        image_path = image_candidate_paths[0]
            
            # Estrategia 2: Si no hay nombre de imagen, buscar por question_id
            if image_path is None:
                img_id = question_id.replace('Q', '').strip()
                candidates_paths = []
                for name in zf.namelist():
                    if not name.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue
                    base = os.path.basename(name)
                    # Intentar coincidencia por ID en el nombre
                    if img_id in base or base.startswith(img_id):
                        candidates_paths.append(name)
                
                if candidates_paths:
                    image_path = candidates_paths[0]
            
            # Estrategia 3: Último recurso - buscar cualquier imagen en el subdirectorio
            if image_path is None:
                for name in zf.namelist():
                    if self.images_subdir.lower() in name.lower() and name.lower().endswith((".jpg", ".jpeg", ".png")):
                        image_path = name
                        break
            
            if image_path is None:
                raise KeyError(
                    f"No se pudo encontrar imagen para {question_id}. "
                    f"Revisa la estructura del ZIP y el mapeo pregunta-imagen."
                )
            
            # Cargar imagen
            with zf.open(image_path) as f:
                image = Image.open(BytesIO(f.read())).convert("RGB")
        finally:
            # Cerrar el zip
            if zf:
                zf.close()
        
        return {
            "image": image,
            "question": question,
            "answer": answer,
            "category": category,
            "candidates": candidates,
            "meta": {
                "question_id": question_id,
                "image_path": image_path,
                "split": self.split
            }
        }

