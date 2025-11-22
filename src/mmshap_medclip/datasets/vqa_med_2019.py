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
    
    Lee archivos del ZIP ImageClef-2019-VQA-Med-Validation.zip:
    - All_QA_Pairs_<split>.txt (contiene preguntas y respuestas)
    - directorio <images_subdir>/ (por defecto Val_images/ para Validation)
    
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
            split: Split a usar ('Validation', 'Test', etc.)
            images_subdir: Subdirectorio dentro del ZIP donde están las imágenes
                          Si es None, se infiere: "Val_images" para Validation, "Test_images" para Test
            n_rows: Número de filas a cargar ("all" o un entero)
        """
        self.zip_path = zip_path
        self.split = split
        
        # Inferir images_subdir si no se proporciona
        if images_subdir is None:
            if split.lower() == "validation":
                self.images_subdir = "Val_images"
            elif split.lower() == "test":
                self.images_subdir = "Test_images"
            else:
                self.images_subdir = f"{split}_images"
        else:
            self.images_subdir = images_subdir
        
        # Cargar preguntas y respuestas desde el ZIP
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Buscar archivo All_QA_Pairs_<split>.txt
            # Para Validation: All_QA_Pairs_val.txt
            # Para Test: All_QA_Pairs_test.txt (probablemente)
            qa_file = None
            split_lower = split.lower()
            
            # Buscar el archivo de QA pairs
            for name in zf.namelist():
                if "All_QA_Pairs" in name and split_lower in name.lower() and name.endswith(".txt"):
                    qa_file = name
                    break
            
            # Si no se encuentra con el split, buscar cualquier All_QA_Pairs
            if qa_file is None:
                for name in zf.namelist():
                    if "All_QA_Pairs" in name and name.endswith(".txt"):
                        qa_file = name
                        break
            
            if qa_file is None:
                raise FileNotFoundError(
                    f"No se encontró All_QA_Pairs_*{split_lower}*.txt en el ZIP. "
                    f"Archivos disponibles: {[n for n in zf.namelist() if '.txt' in n][:10]}"
                )
            
            # Leer archivo de QA pairs
            # Formato esperado: QID\tQuestion\tAnswer\t[ImageFilename] (o variaciones)
            self.samples = []
            image_map = {}  # question_id -> image_filename
            
            with zf.open(qa_file) as f:
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
                    category = self._infer_category(question)
                    
                    self.samples.append({
                        'question_id': q_id,
                        'question': question,
                        'answer': answer,
                        'category': category,
                        'image_filename': image_map.get(q_id)  # Puede ser None
                    })
            
            # Construir índice de imágenes (basename -> ruta completa)
            self._name_to_path = {}
            for name in zf.namelist():
                if name.endswith("/") or not name.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                base = os.path.basename(name)
                # Priorizar imágenes en el subdirectorio correcto
                score = int(self.images_subdir.lower() in name.lower())
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

