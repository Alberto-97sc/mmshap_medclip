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
    - VQAMed2019_<split>_Questions.txt
    - VQAMed2019_<split>_Answers.txt
    - directorio Images/
    
    Infiere categorías de preguntas y construye candidatos automáticamente.
    """
    
    def __init__(
        self,
        zip_path: str,
        split: str,
        images_subdir: str = "Images",
        n_rows: str = "all"
    ):
        """
        Args:
            zip_path: Ruta al archivo ZIP del dataset
            split: Split a usar ('Validation', 'Test', etc.)
            images_subdir: Subdirectorio dentro del ZIP donde están las imágenes
            n_rows: Número de filas a cargar ("all" o un entero)
        """
        self.zip_path = zip_path
        self.split = split
        self.images_subdir = images_subdir
        
        # Cargar preguntas y respuestas desde el ZIP
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Buscar archivos de preguntas y respuestas
            questions_file = None
            answers_file = None
            
            for name in zf.namelist():
                if f"VQAMed2019_{split}_Questions.txt" in name:
                    questions_file = name
                elif f"VQAMed2019_{split}_Answers.txt" in name:
                    answers_file = name
            
            if questions_file is None:
                raise FileNotFoundError(
                    f"No se encontró VQAMed2019_{split}_Questions.txt en el ZIP"
                )
            if answers_file is None:
                raise FileNotFoundError(
                    f"No se encontró VQAMed2019_{split}_Answers.txt en el ZIP"
                )
            
            # Leer preguntas
            # Formato típico: Q1\timage_name.jpg\tquestion text
            # o Q1\tquestion text (sin imagen)
            questions_dict = {}
            image_map = {}  # question_id -> image_filename
            
            with zf.open(questions_file) as f:
                for line in f:
                    line = line.decode('utf-8').strip()
                    if not line:
                        continue
                    # Intentar separar por tab (formato más común)
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        q_id = parts[0].strip()
                        # Si hay 3 partes: Q1, imagen, pregunta
                        if len(parts) == 3:
                            image_map[q_id] = parts[1].strip()
                            question = parts[2].strip()
                        else:
                            # Si hay 2 partes: Q1, pregunta
                            question = parts[1].strip()
                    else:
                        # Intentar separar por espacio si no hay tab
                        parts = line.split(' ', 1)
                        if len(parts) == 2:
                            q_id = parts[0].strip()
                            question = parts[1].strip()
                        else:
                            continue
                    questions_dict[q_id] = question
            
            # Leer respuestas
            answers_dict = {}
            with zf.open(answers_file) as f:
                for line in f:
                    line = line.decode('utf-8').strip()
                    if not line:
                        continue
                    # Formato esperado: Q1\tanswer text o Q1 answer text
                    parts = line.split('\t', 1)
                    if len(parts) == 2:
                        q_id, answer = parts
                    else:
                        parts = line.split(' ', 1)
                        if len(parts) == 2:
                            q_id, answer = parts
                        else:
                            continue
                    answers_dict[q_id.strip()] = answer.strip()
            
            # Emparejar preguntas y respuestas
            self.samples = []
            for q_id in questions_dict:
                if q_id in answers_dict:
                    question = questions_dict[q_id]
                    answer = answers_dict[q_id]
                    
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
                score = int(images_subdir.lower() in name.lower())
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
        
        Categorías:
        - "modality" → modality
        - "plane" → plane
        - "organ system" → organ_system
        - "abnormality" → abnormality
        """
        question_lower = question.lower()
        
        # Buscar palabras clave para cada categoría
        if "modality" in question_lower:
            return "modality"
        elif "plane" in question_lower:
            return "plane"
        elif "organ system" in question_lower or "organ" in question_lower:
            return "organ_system"
        elif "abnormality" in question_lower or "abnormal" in question_lower:
            return "abnormality"
        else:
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

