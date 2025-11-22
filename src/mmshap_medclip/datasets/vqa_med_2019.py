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
            zip_path: Ruta al archivo ZIP del dataset
            split: Split a usar ('Training', 'Validation', 'Test', etc.)
            images_subdir: Subdirectorio dentro del ZIP donde están las imágenes
                          Si es None, se infiere: "Train_images" para Training, "Val_images" para Validation, "Test_images" para Test
            n_rows: Número de filas a cargar ("all" o un entero)
        """
        self.zip_path = zip_path
        self.split = split
        
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
        
        # Detectar el prefijo de directorio raíz del ZIP si existe
        # Por ejemplo: "ImageClef-2019-VQA-Med-Validation/"
        self.zip_root_prefix = None
        
        # Cargar preguntas y respuestas desde el ZIP
        with zipfile.ZipFile(zip_path, "r") as zf:
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
            
            # Estrategia 4: Buscar archivos por categoría y combinarlos
            # Si no hay All_QA_Pairs, buscar archivos C1_Modality_train.txt, C2_Plane_val.txt, etc.
            category_files = []
            if qa_file is None:
                for name in all_txt_files:
                    basename = os.path.basename(name)
                    # Buscar archivos de categoría: C1_Modality_train.txt, C2_Plane_val.txt, etc.
                    # Buscar por split (train, val, test) o por número de categoría
                    if (basename.startswith("C") and 
                        any(cat in basename.lower() for cat in ["modality", "plane", "organ", "abnormality"])):
                        # Verificar si coincide con el split (train/val/test)
                        basename_lower = basename.lower()
                        split_match = (
                            split_lower in basename_lower or
                            ("train" in basename_lower and split_lower in ["training", "train"]) or
                            ("val" in basename_lower and split_lower == "validation") or
                            ("test" in basename_lower and split_lower == "test")
                        )
                        if split_match:
                            category_files.append(name)
                
                if category_files:
                    # Si encontramos archivos por categoría, los usaremos más adelante
                    # Por ahora, usamos el primero como referencia
                    qa_file = category_files[0]
                    print(f"⚠️  No se encontró All_QA_Pairs, usando archivos por categoría: {category_files}")
            
            if qa_file is None:
                # Mostrar todos los archivos disponibles para debugging
                txt_files = [n for n in zf.namelist() if n.endswith(".txt")]
                # Mostrar también estructura de directorios
                dirs = sorted(set([os.path.dirname(n) for n in zf.namelist() if os.path.dirname(n)]))
                
                error_msg = (
                    f"No se encontró archivo All_QA_Pairs_*{split_lower}*.txt en el ZIP.\n"
                    f"Archivos .txt disponibles ({len(txt_files)}):\n" +
                    "\n".join(f"  - {f}" for f in txt_files[:20]) +
                    (f"\n  ... y {len(txt_files) - 20} más" if len(txt_files) > 20 else "") +
                    f"\n\nDirectorios en el ZIP ({len(dirs)}):\n" +
                    "\n".join(f"  - {d}" for d in dirs[:10]) +
                    (f"\n  ... y {len(dirs) - 10} más" if len(dirs) > 10 else "")
                )
                raise FileNotFoundError(error_msg)
            
            # Leer archivo(s) de QA pairs
            # Si encontramos archivos por categoría, leer todos y combinarlos
            files_to_read = [qa_file]
            if category_files and qa_file in category_files:
                # Si qa_file es uno de los archivos de categoría, leer todos
                files_to_read = category_files
            
            # Formato esperado: QID\tQuestion\tAnswer\t[ImageFilename] (o variaciones)
            self.samples = []
            image_map = {}  # question_id -> image_filename
            
            for file_to_read in files_to_read:
                with zf.open(file_to_read) as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.decode('utf-8').strip()
                        if not line:
                            continue
                        
                        # Intentar separar por tab primero (formato más común en datasets VQA)
                        parts = line.split('\t')
                        
                        q_id = None
                        question = None
                        answer = None
                        image_filename = None
                        
                        if len(parts) >= 3:
                            # Formato: QID\tQuestion\tAnswer\t[ImageFilename]
                            q_id = parts[0].strip()
                            question = parts[1].strip()
                            answer = parts[2].strip()
                            
                            # Si hay 4 o más partes, la última puede ser el nombre de imagen
                            if len(parts) >= 4:
                                image_filename = parts[3].strip()
                        elif len(parts) == 2:
                            # Formato alternativo: QID\tQuestion|Answer
                            q_id = parts[0].strip()
                            rest = parts[1].strip()
                            # Intentar separar pregunta y respuesta por '|' o por posición de '?'
                            if '|' in rest:
                                question, answer = rest.split('|', 1)
                                question = question.strip()
                                answer = answer.strip()
                            elif '?' in rest:
                                # Si hay signo de interrogación, separar ahí
                                q_parts = rest.split('?', 1)
                                question = q_parts[0].strip() + '?'
                                answer = q_parts[1].strip() if len(q_parts) > 1 else "unknown"
                            else:
                                # Si no hay separador claro, asumir que todo es pregunta
                                question = rest
                                answer = "unknown"
                        elif len(parts) == 1:
                            # Formato con espacios o sin separadores claros
                            # Intentar separar por espacios: QID Question Answer
                            parts_space = line.split(None, 2)  # Máximo 3 partes
                            if len(parts_space) >= 3:
                                q_id = parts_space[0].strip()
                                question = parts_space[1].strip()
                                answer = parts_space[2].strip()
                            elif len(parts_space) == 2:
                                q_id = parts_space[0].strip()
                                rest = parts_space[1].strip()
                                if '?' in rest:
                                    q_parts = rest.split('?', 1)
                                    question = q_parts[0].strip() + '?'
                                    answer = q_parts[1].strip() if len(q_parts) > 1 else "unknown"
                                else:
                                    question = rest
                                    answer = "unknown"
                        
                        # Validar que tenemos los campos mínimos
                        if not q_id or not question or not answer:
                            # Saltar líneas que no se pueden parsear
                            if line_num <= 5:  # Solo mostrar advertencia para las primeras líneas
                                print(f"⚠️  Advertencia: No se pudo parsear la línea {line_num}: {line[:80]}")
                            continue
                        
                        # Guardar imagen si se encontró
                        if image_filename:
                            image_map[q_id] = image_filename
                        
                        # Inferir categoría de la pregunta
                        # Si estamos leyendo archivos por categoría, intentar inferir desde el nombre del archivo
                        category = self._infer_category(question)
                        if category_files and file_to_read in category_files:
                            # Intentar inferir categoría desde el nombre del archivo
                            basename = os.path.basename(file_to_read).lower()
                            if "modality" in basename or "c1" in basename:
                                category = "modality"
                            elif "plane" in basename or "c2" in basename:
                                category = "plane"
                            elif "organ" in basename or "c3" in basename:
                                category = "organ_system"
                            elif "abnormality" in basename or "c4" in basename:
                                category = "abnormality"
                        
                        self.samples.append({
                            'question_id': q_id,
                            'question': question,
                            'answer': answer,
                            'category': category,
                            'image_filename': image_map.get(q_id)  # Puede ser None
                        })
            
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
            
            # Construir candidatos por categoría
            self._candidates_by_category = self._build_candidates_by_category()
            
            # Limitar número de muestras si se especifica
            if n_rows != "all":
                self.samples = self.samples[:int(n_rows)]
    
    def _infer_category(self, question: str) -> str:
        """
        Infiere la categoría de la pregunta analizando su texto.
        
        Categorías según VQA-Med 2019:
        - C1_Modality → modality
        - C2_Plane → plane
        - C3_Organ → organ_system
        - C4_Abnormality → abnormality
        """
        question_lower = question.lower()
        
        # Buscar palabras clave para cada categoría
        # Modality: tipo de modalidad (X-ray, CT, MRI, etc.)
        modality_keywords = ["modality", "x-ray", "xray", "ct scan", "mri", "ultrasound", "pet", "spect"]
        if any(kw in question_lower for kw in modality_keywords):
            return "modality"
        
        # Plane: plano de la imagen (axial, sagittal, coronal, etc.)
        plane_keywords = ["plane", "axial", "sagittal", "coronal", "transverse", "frontal"]
        if any(kw in question_lower for kw in plane_keywords):
            return "plane"
        
        # Organ: sistema de órganos
        organ_keywords = ["organ", "system", "lung", "heart", "liver", "kidney", "brain", "chest", "abdomen"]
        if any(kw in question_lower for kw in organ_keywords):
            return "organ_system"
        
        # Abnormality: anormalidades o patologías
        abnormality_keywords = ["abnormality", "abnormal", "disease", "pathology", "lesion", "tumor", "fracture", "pneumonia"]
        if any(kw in question_lower for kw in abnormality_keywords):
            return "abnormality"
        
        # Categoría por defecto si no se puede inferir
        return "other"
    
    def _build_candidates_by_category(self) -> Dict[str, List[str]]:
        """
        Construye la lista de candidatos válidos por categoría.
        Todas las respuestas únicas de esa categoría dentro del split.
        """
        candidates_by_category = defaultdict(set)
        
        for sample in self.samples:
            category = sample['category']
            answer = sample['answer']
            candidates_by_category[category].add(answer)
        
        # Convertir sets a listas ordenadas
        return {
            category: sorted(list(answers))
            for category, answers in candidates_by_category.items()
        }
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        question = sample['question']
        answer = sample['answer']
        category = sample['category']
        question_id = sample['question_id']
        
        # Obtener candidatos para esta categoría
        candidates = self._candidates_by_category.get(category, [])
        
        # Intentar encontrar la imagen asociada
        image_path = None
        image_filename = sample.get('image_filename')
        
        with zipfile.ZipFile(self.zip_path, "r") as zf:
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

