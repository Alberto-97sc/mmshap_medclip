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
    
    Lee archivos del ZIP ImageClef-2019-VQA-Med-Training.zip o Validation.zip:
    - All_QA_Pairs_<split>.txt (contiene preguntas y respuestas)
    - directorio <images_subdir>/ (por defecto Train_images/ para Training, Val_images/ para Validation)
    
    También puede leer desde el ZIP padre (VQA-Med-2019.zip) que contiene los zips hijos.
    
    Infiere categorías de preguntas y construye candidatos automáticamente.
    """
    
    def __init__(
        self,
        zip_path: str,
        split: str = "Validation",
        images_subdir: str = None,
        n_rows: str = "all"
    ):
        """
        Args:
            zip_path: Ruta al archivo ZIP del dataset (puede ser el zip padre VQA-Med-2019.zip o el zip hijo)
            split: Split a usar ('Training', 'Validation', 'Test', etc.)
            images_subdir: Subdirectorio dentro del ZIP donde están las imágenes
                          Si es None, se infiere: "Train_images" para Training, "Val_images" para Validation, "Test_images" para Test
            n_rows: Número de filas a cargar ("all" o un entero)
        """
        self.zip_path = zip_path
        self.split = split
        
        # Inicializar candidates_per_cat como dict vacío
        self.candidates_per_cat = {}
        
        # Inferir images_subdir si no se proporciona
        # Nota: El ZIP puede tener un directorio raíz, así que buscamos en cualquier ubicación
        if images_subdir is None:
            if split.lower() == "validation":
                self.images_subdir = "Val_images"
            elif split.lower() == "training" or split.lower() == "train":
                self.images_subdir = "Train_images"
            elif split.lower() == "test":
                self.images_subdir = "Test_images"
            else:
                self.images_subdir = f"{split}_images"
        else:
            self.images_subdir = images_subdir
        
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
            # Determinar el nombre del zip hijo según el split
            if split.lower() in ["training", "train"]:
                self.inner_zip_name = "ImageClef-2019-VQA-Med-Training.zip"
            elif split.lower() == "validation":
                self.inner_zip_name = "ImageClef-2019-VQA-Med-Validation.zip"
            elif split.lower() == "test":
                self.inner_zip_name = "VQAMed2019Test.zip"
            else:
                # Intentar inferir desde el nombre del zip
                self.inner_zip_name = f"ImageClef-2019-VQA-Med-{split}.zip"
        
        # Detectar el prefijo de directorio raíz del ZIP si existe
        # Por ejemplo: "ImageClef-2019-VQA-Med-Validation/"
        self.zip_root_prefix = None
        
        # Identificar el split desde el nombre del directorio/archivo
        # Si el path contiene "ImageClef-2019-VQA-Med-Training" → usar solo *train.txt
        # Si contiene "ImageClef-2019-VQA-Med-Validation" → usar solo *val.txt
        self.detected_split = None
        split_lower = split.lower()
        if split_lower in ["training", "train"]:
            self.detected_split = "train"
        elif split_lower == "validation":
            self.detected_split = "val"
        elif split_lower == "test":
            self.detected_split = "test"
        else:
            # Intentar inferir desde el split
            if "train" in split_lower:
                self.detected_split = "train"
            elif "val" in split_lower:
                self.detected_split = "val"
            elif "test" in split_lower:
                self.detected_split = "test"
        
        if self.detected_split is None:
            raise ValueError(f"No se pudo identificar el split desde '{split}'. Debe ser 'Training', 'Validation' o 'Test'")
        
        print(f"📊 Split detectado: '{split}' → '{self.detected_split}' (usará solo archivos *{self.detected_split}.txt)")
        
        # Cargar preguntas y respuestas desde el ZIP
        # Si es un zip anidado, abrir el zip padre y luego el zip hijo
        if self.is_nested_zip:
            # Abrir el zip padre
            with zipfile.ZipFile(zip_path, "r") as parent_zip:
                # Verificar que el zip hijo existe
                if self.inner_zip_name not in parent_zip.namelist():
                    # Buscar con variaciones
                    found = False
                    for name in parent_zip.namelist():
                        if split.lower() in name.lower() and name.endswith(".zip"):
                            self.inner_zip_name = name
                            found = True
                            break
                    if not found:
                        raise FileNotFoundError(
                            f"No se encontró el zip hijo para split '{split}' en {zip_path}. "
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
            # Buscar archivo All_QA_Pairs_<split>.txt
            # Para Training: All_QA_Pairs_train.txt
            # Para Validation: All_QA_Pairs_val.txt
            # Para Test: All_QA_Pairs_test.txt
            qa_file = None
            split_lower = split.lower()
            
            # Lista de nombres posibles para el archivo (buscar en cualquier ubicación)
            possible_names = [
                f"All_QA_Pairs_{split_lower}.txt",
                f"All_QA_Pairs_val.txt",  # Para Validation
                f"All_QA_Pairs_train.txt",  # Para Training
                f"All_QA_Pairs_test.txt",  # Para Test
                "All_QA_Pairs_val.txt",
                "All_QA_Pairs_train.txt",
                "All_QA_Pairs.txt",
            ]
            
            # También buscar en subdirectorios
            all_txt_files = [n for n in zf.namelist() if n.endswith(".txt")]
            all_files = zf.namelist()  # Todos los archivos para debugging
            
            # Detectar prefijo de directorio raíz del ZIP (ej: "ImageClef-2019-VQA-Med-Validation/")
            # Buscar el directorio más común en las rutas
            if all_files:
                # Obtener el primer directorio común
                first_file = all_files[0]
                if '/' in first_file:
                    # Extraer el prefijo del directorio raíz
                    parts = first_file.split('/')
                    if len(parts) > 1:
                        self.zip_root_prefix = parts[0] + '/'
                        print(f"📂 Detectado prefijo de directorio en ZIP: {self.zip_root_prefix}")
            
            # PRIORIDAD 1: Buscar archivos QAPairsByCategory (C1_Modality_*, C2_Plane_*, C3_Organ_*)
            # IGNORAR C4_Abnormality_* completamente
            # FILTRAR ESTRICTAMENTE por split: solo *train.txt para Training, *val.txt para Validation
            # Estos archivos tienen prioridad sobre All_QA_Pairs
            category_files = []
            for name in all_txt_files:
                basename = os.path.basename(name)
                # Buscar archivos de categoría: C1_Modality_train.txt, C2_Plane_val.txt, etc.
                # Verificar formato C1_*, C2_*, C3_* (IGNORAR C4_*)
                if basename.startswith("C") and len(basename) > 1:
                    # IGNORAR archivos de abnormality (C4_*)
                    basename_lower = basename.lower()
                    if "c4" in basename_lower or "abnormality" in basename_lower:
                        continue  # Saltar archivos de abnormality
                    
                    # FILTRAR ESTRICTAMENTE por split: debe terminar en *{detected_split}.txt
                    # NO mezclar train y val bajo ningún criterio
                    if not basename_lower.endswith(f"{self.detected_split}.txt"):
                        continue  # Saltar archivos que no coinciden con el split
                    
                    # Verificar que el archivo pertenece al split correcto
                    # Asegurar que contiene el sufijo del split en el nombre
                    if f"_{self.detected_split}.txt" in basename_lower or f"-{self.detected_split}.txt" in basename_lower:
                        category_files.append(name)
                    else:
                        # Log para debugging si hay archivos que casi coinciden
                        pass  # Silenciosamente saltar archivos que no coinciden con el split
            
            # Si encontramos archivos por categoría, usarlos (tienen prioridad)
            if category_files:
                qa_file = None  # Forzar uso de archivos por categoría
                print(f"📁 Usando archivos por categoría: {len(category_files)} archivos encontrados")
                print(f"   Archivos: {[os.path.basename(f) for f in category_files]}")
            else:
                # Estrategia 1: Buscar por nombre exacto (con y sin prefijo de directorio)
                # Buscar tanto en raíz como en subdirectorios
                for name in all_txt_files:
                    basename = os.path.basename(name)
                    if basename in possible_names:
                        qa_file = name
                        break
                
                # Estrategia 2: Buscar por patrón "All_QA_Pairs" + split (en cualquier ubicación)
                if qa_file is None:
                    for name in all_txt_files:
                        basename = os.path.basename(name)
                        if "All_QA_Pairs" in basename and split_lower in basename.lower():
                            qa_file = name
                            break
                
                # Estrategia 3: Buscar cualquier archivo con "All_QA_Pairs" (en cualquier ubicación)
                if qa_file is None:
                    for name in all_txt_files:
                        basename = os.path.basename(name)
                        if "All_QA_Pairs" in basename:
                            qa_file = name
                            break
            
            # Si no encontramos archivos por categoría ni All_QA_Pairs, mostrar error
            if not category_files and qa_file is None:
                # Mostrar todos los archivos disponibles para debugging
                txt_files = [n for n in zf.namelist() if n.endswith(".txt")]
                # Mostrar también estructura de directorios
                dirs = sorted(set([os.path.dirname(n) for n in zf.namelist() if os.path.dirname(n)]))
                
                error_msg = (
                    f"No se encontraron archivos de QA pairs para split '{split}' en el ZIP.\n"
                    f"Buscando archivos QAPairsByCategory (C1_*, C2_*, C3_*, C4_*) o All_QA_Pairs_*{split_lower}*.txt\n"
                    f"Archivos .txt disponibles ({len(txt_files)}):\n" +
                    "\n".join(f"  - {f}" for f in txt_files[:20]) +
                    (f"\n  ... y {len(txt_files) - 20} más" if len(txt_files) > 20 else "") +
                    f"\n\nDirectorios en el ZIP ({len(dirs)}):\n" +
                    "\n".join(f"  - {d}" for d in dirs[:10]) +
                    (f"\n  ... y {len(dirs) - 10} más" if len(dirs) > 10 else "")
                )
                raise FileNotFoundError(error_msg)
            
            # Leer archivo(s) de QA pairs
            # IMPORTANTE: Solo usar archivos que coincidan con el split detectado
            # NO mezclar train y val bajo ningún criterio
            files_to_read = []
            
            if category_files:
                # category_files ya está filtrado por split, pero verificamos nuevamente por seguridad
                for f in category_files:
                    basename_lower = os.path.basename(f).lower()
                    # Verificar que termina en {detected_split}.txt
                    if basename_lower.endswith(f"{self.detected_split}.txt"):
                        files_to_read.append(f)
                    else:
                        print(f"⚠️  Saltando archivo '{os.path.basename(f)}' (no termina en '{self.detected_split}.txt')")
            elif qa_file:
                # Verificar que qa_file también coincida con el split
                basename_lower = os.path.basename(qa_file).lower()
                if self.detected_split in basename_lower:
                    files_to_read = [qa_file]
                else:
                    print(f"⚠️  Archivo All_QA_Pairs '{os.path.basename(qa_file)}' no coincide con split '{self.detected_split}'")
            
            if not files_to_read:
                raise FileNotFoundError(
                    f"No se encontraron archivos de QA pairs para split '{self.detected_split}' (split original: '{split}'). "
                    f"Archivos encontrados pero filtrados: {len(category_files) if category_files else 0} archivos de categoría"
                )
            
            print(f"📁 Archivos a leer para split '{self.detected_split}': {len(files_to_read)} archivos")
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
                # Buscar "Val_images" o "images_subdir" en cualquier parte de la ruta
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
            print(f"📊 Construyendo candidatos desde {len(self.samples)} muestras del split '{self.detected_split}'...")
            self._build_candidates_by_category()
            
            # Limitar número de muestras si se especifica
            if n_rows != "all":
                self.samples = self.samples[:int(n_rows)]
                # Reconstruir candidatos después de limitar muestras
                print(f"📊 Reconstruyendo candidatos después de limitar a {n_rows} muestras...")
                self._build_candidates_by_category()
            
            # Verificar que todas las muestras tienen categorías válidas con candidatos
            # ANTES de devolver cualquier muestra, verificar que sample["category"] exista en candidates_per_cat
            samples_to_remove = []
            for idx, sample in enumerate(self.samples):
                category = sample.get('category')
                if category not in self.candidates_per_cat:
                    print(f"⚠️  ADVERTENCIA: Muestra {idx} tiene categoría '{category}' que no existe en candidates_per_cat")
                    print(f"   - image_id: {sample.get('question_id', 'N/A')}")
                    print(f"   - question: {sample.get('question', 'N/A')[:80]}...")
                    print(f"   - answer: {sample.get('answer', 'N/A')}")
                    print(f"   Categorías disponibles: {sorted(self.candidates_per_cat.keys())}")
                    samples_to_remove.append(idx)
            
            # Remover muestras sin categorías válidas (en orden inverso para no afectar índices)
            if samples_to_remove:
                print(f"⚠️  Removiendo {len(samples_to_remove)} muestras sin categorías válidas...")
                for idx in reversed(samples_to_remove):
                    self.samples.pop(idx)
                # Reconstruir candidatos después de remover muestras inválidas
                self._build_candidates_by_category()
                print(f"✅ Dataset final: {len(self.samples)} muestras válidas")
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
        
        # Debug: mostrar estadísticas
        print(f"📊 Construyendo candidatos desde {len(self.samples)} muestras...")
        if self.candidates_per_cat:
            print(f"📊 Candidatos construidos por categoría:")
            for cat, cands in self.candidates_per_cat.items():
                print(f"   {cat}: {len(cands)} candidatos")
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
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        question = sample['question']
        answer = sample['answer']
        # SIEMPRE asignar category y candidates como se especifica
        # La categoría ya debería estar normalizada al construir samples
        category = sample.get('category', '').strip()
        question_id = sample.get('question_id', '')
        image_filename = sample.get('image_filename', '')
        
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
            # Descartar esta muestra con un warning (no debería llegar aquí si el filtrado funcionó)
            raise ValueError(
                f"Muestra {idx} (image_id={question_id}) tiene categoría '{category}' que no existe en candidates_per_cat. "
                f"Esta muestra debería haber sido filtrada durante la construcción del dataset. "
                f"Categorías disponibles: {sorted(self.candidates_per_cat.keys())}"
            )
        
        # Obtener candidatos para esta categoría (no globales, solo de esta categoría)
        candidates = self.candidates_per_cat.get(category, [])
        
        # Verificación final: si no hay candidatos, esto es un error crítico
        if not candidates:
            print(f"⚠️  ADVERTENCIA CRÍTICA: No se encontraron candidatos para categoría '{category}'")
            print(f"   Muestra idx={idx}:")
            print(f"   - image_id: {question_id}")
            print(f"   - question: {question[:80]}...")
            print(f"   - category: '{category}'")
            print(f"   - answer: {answer}")
            print(f"   Categorías disponibles en candidates_per_cat: {sorted(self.candidates_per_cat.keys())}")
            print(f"   Total muestras en dataset: {len(self.samples)}")
            # Esta muestra no debería estar en el dataset si no tiene candidatos
            # Lanzar error para que el código que llama pueda manejarlo
            raise ValueError(
                f"Muestra {idx} (image_id={question_id}) tiene categoría '{category}' sin candidatos. "
                f"Esto indica un problema en la construcción del dataset. "
                f"Categorías disponibles: {sorted(self.candidates_per_cat.keys())}"
            )
        
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
                    candidates = [n for n in zf.namelist() 
                                if n.endswith(image_filename) or os.path.basename(n) == image_filename]
                    if candidates:
                        image_path = candidates[0]
            
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

