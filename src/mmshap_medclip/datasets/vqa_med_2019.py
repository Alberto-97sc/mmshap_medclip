import os
import zipfile
from PIL import Image
from io import BytesIO
from typing import Dict, List, Optional
from collections import defaultdict
from mmshap_medclip.datasets.base import DatasetBase
from mmshap_medclip.registry import register_dataset

@register_dataset("vqa_med_2019")
def _build_vqa_med_2019(params):
    return VQAMed2019Dataset(**params)


class VQAMed2019Dataset(DatasetBase):
    """
    Dataset loader para VQA-Med 2019.

    Soporta los splits oficiales: train, validation y test.
    """

    VALID_CATEGORIES = {"modality", "plane", "organ_system"}

    SPLIT_ALIASES = {
        "train": {"train", "training"},
        "val": {"val", "validation", "dev"},
        "test": {"test", "testing"},
    }

    SPLIT_LABELS = {
        "train": "Training",
        "val": "Validation",
        "test": "Test",
    }

    SPLIT_SUFFIXES = {
        "train": ("train", "training"),
        "val": ("val", "validation"),
        "test": ("test",),
    }

    INNER_ZIP_TOKENS = {
        "train": ("train", "training"),
        "val": ("val", "validation"),
        "test": ("test",),
    }

    DEFAULT_IMAGE_SUBDIR = {
        "train": "Train_images",
        "val": "Val_images",
        "test": "VQAMed2019_Test_Images",
    }

    def __init__(
        self,
        zip_path: str,
        split: str = "Training",
        images_subdir: Optional[str] = None,
        n_rows: str = "all",
    ):
        self.zip_path = zip_path
        self.split_key = self._normalize_split(split)
        self.split = self.split_key
        self.split_label = self.SPLIT_LABELS[self.split_key]
        print(f"📊 Split seleccionado: {self.split_label.upper()}")

        self.images_subdir = images_subdir or self.DEFAULT_IMAGE_SUBDIR.get(self.split_key)
        self.n_rows = n_rows

        self.samples: List[dict] = []
        self.candidates_per_cat: Dict[str, List[str]] = {}
        self.zip_root_prefix: Optional[str] = None
        self._name_to_path: Dict[str, str] = {}
        self._test_images_zip_data: Optional[bytes] = None

        self.is_nested_zip = False
        self.inner_zip_name: Optional[str] = None
        self.inner_zip_data: Optional[bytes] = None

        self._prepare_zip_handle()
        self._load_split_contents()
        self._finalize_samples()

    # ------------------------------------------------------------------
    # Inicialización y utilidades de ZIP
    # ------------------------------------------------------------------
    def _normalize_split(self, split: str) -> str:
        split_lower = (split or "train").lower()
        for canonical, aliases in self.SPLIT_ALIASES.items():
            if split_lower in aliases:
                return canonical
        raise ValueError(
            f"Split '{split}' no soportado. Usa uno de: {', '.join(self.SPLIT_LABELS.values())}."
        )

    def _prepare_zip_handle(self) -> None:
        zip_basename = os.path.basename(self.zip_path).lower()
        if not zip_basename.endswith(".zip"):
            raise ValueError(f"zip_path debe ser un archivo ZIP, recibido: {self.zip_path}")

        split_tokens = self.INNER_ZIP_TOKENS[self.split_key]
        contains_split = any(token in zip_basename for token in split_tokens)
        is_parent_zip = "vqa-med-2019" in zip_basename and not contains_split

        if is_parent_zip:
            self.is_nested_zip = True
            with zipfile.ZipFile(self.zip_path, "r") as parent_zip:
                inner_name = self._find_inner_zip_name(parent_zip.namelist())
                self.inner_zip_name = inner_name
                self.inner_zip_data = parent_zip.read(inner_name)
        else:
            self.is_nested_zip = False

    def _find_inner_zip_name(self, names: List[str]) -> str:
        tokens = self.INNER_ZIP_TOKENS[self.split_key]
        for name in names:
            lower = name.lower()
            if "__macosx" in lower or not lower.endswith(".zip"):
                continue
            if any(token in lower for token in tokens):
                return name
        raise FileNotFoundError(
            f"No se encontró un ZIP interno para el split {self.split_label} dentro del archivo proporcionado."
        )

    def _open_primary_zip(self):
        if self.is_nested_zip:
            return zipfile.ZipFile(BytesIO(self.inner_zip_data), "r")
        return zipfile.ZipFile(self.zip_path, "r")

    def _open_test_images_zip(self):
        if not self._test_images_zip_data:
            raise RuntimeError("No se inicializó el ZIP de imágenes para el split TEST.")
        return zipfile.ZipFile(BytesIO(self._test_images_zip_data), "r")

    def _detect_root_prefix(self, names: List[str]) -> Optional[str]:
        if not names:
            return None
        first = names[0]
        if "/" in first:
            prefix = first.split("/", 1)[0] + "/"
            print(f"📂 Detectado prefijo de directorio en ZIP: {prefix}")
            return prefix
        return None

    # ------------------------------------------------------------------
    # Carga de datos según split
    # ------------------------------------------------------------------
    def _load_split_contents(self) -> None:
        with self._open_primary_zip() as zf:
            self.zip_root_prefix = self._detect_root_prefix(zf.namelist())
            if self.split_key in ("train", "val"):
                self._load_train_val_split(zf)
            else:
                self._load_test_split(zf)

    def _load_train_val_split(self, zf: zipfile.ZipFile) -> None:
        suffixes = self.SPLIT_SUFFIXES[self.split_key]
        split_label = self.split_label.upper()
        all_txt_files = [n for n in zf.namelist() if n.lower().endswith(".txt")]

        category_files = []
        for name in all_txt_files:
            dirname = os.path.dirname(name).lower()
            basename = os.path.basename(name)
            basename_lower = basename.lower()

            if "qapairsbycategory" not in dirname:
                continue
            if not basename_lower.startswith("c"):
                continue
            if "c4" in basename_lower or "abnormality" in basename_lower:
                continue
            if not self._filename_matches_split(basename_lower, suffixes):
                continue

            category_files.append(name)

        if not category_files:
            txt_files = [n for n in zf.namelist() if n.lower().endswith(".txt")]
            dirs = sorted(set(os.path.dirname(n) for n in zf.namelist() if os.path.dirname(n)))
            raise FileNotFoundError(
                f"No se encontraron archivos QAPairsByCategory para el split {split_label}.\n"
                f"Se esperaban archivos C1/C2/C3 *_{suffixes[0]}.txt dentro de QAPairsByCategory/.\n"
                f"Archivos .txt disponibles ({len(txt_files)}):\n" +
                "\n".join(f"  - {f}" for f in txt_files[:20]) +
                (f"\n  ... y {len(txt_files) - 20} más" if len(txt_files) > 20 else "") +
                f"\n\nDirectorios en el ZIP ({len(dirs)}):\n" +
                "\n".join(f"  - {d}" for d in dirs[:10]) +
                (f"\n  ... y {len(dirs) - 10} más" if len(dirs) > 10 else "")
            )

        category_files = sorted(category_files)
        print(f"📁 Archivos a leer para split {split_label}: {len(category_files)} archivos")
        for f in category_files:
            print(f"   - {os.path.basename(f)}")

        for file_to_read in category_files:
            category = self._infer_category_from_filename(file_to_read)
            if category == "abnormality":
                continue

            with zf.open(file_to_read) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.decode("utf-8").strip()
                    if not line:
                        continue

                    parts = line.split("|")
                    if len(parts) != 3:
                        if line_num <= 5:
                            print(f"⚠️  Línea {line_num} inválida en {file_to_read}: {line[:80]}")
                        continue

                    image_id, question, answer = [p.strip() for p in parts]
                    if not image_id or not question or not answer:
                        if line_num <= 5:
                            print(f"⚠️  Campos vacíos en línea {line_num}: {line[:80]}")
                        continue

                    self.samples.append({
                        "question_id": image_id,
                        "question": question,
                        "answer": answer,
                        "category": category,
                        "image_filename": image_id,
                    })

        self._build_image_index_from_zip(zf, self.zip_root_prefix)

    def _load_test_split(self, zf: zipfile.ZipFile) -> None:
        names = zf.namelist()
        qa_file = self._find_test_qa_file(names)
        print(f"📁 Archivo QA (TEST): {qa_file}")

        with zf.open(qa_file) as f:
            for line_num, line in enumerate(f, 1):
                line = line.decode("utf-8").strip()
                if not line:
                    continue

                parts = line.split("|")
                if len(parts) < 4:
                    if line_num <= 5:
                        print(f"⚠️  Línea {line_num} sin formato esperado en TEST: {line[:80]}")
                    continue

                image_id = parts[0].strip()
                category = parts[1].strip().lower()
                question = parts[2].strip()
                answer = parts[3].strip()

                if category not in self.VALID_CATEGORIES:
                    continue

                self.samples.append({
                    "question_id": image_id,
                    "question": question,
                    "answer": answer,
                    "category": category,
                    "image_filename": image_id,
                })

        image_zip_name = self._find_test_images_zip(names)
        print(f"📁 ZIP de imágenes (TEST): {image_zip_name}")
        self._test_images_zip_data = zf.read(image_zip_name)

        with zipfile.ZipFile(BytesIO(self._test_images_zip_data), "r") as img_zip:
            prefix = self._detect_root_prefix(img_zip.namelist())
            self._build_image_index_from_zip(img_zip, prefix)

    def _filename_matches_split(self, basename_lower: str, suffixes: tuple) -> bool:
        for suffix in suffixes:
            suffix = suffix.lower()
            if basename_lower.endswith(f"_{suffix}.txt") or basename_lower.endswith(f"-{suffix}.txt"):
                return True
            if basename_lower.endswith(f"{suffix}.txt") and suffix not in {"val", "test"}:
                return True
        return False

    def _find_test_qa_file(self, names: List[str]) -> str:
        for name in names:
            lower = name.lower()
            if "__macosx" in lower:
                continue
            if lower.endswith(".txt") and "question" in lower and "answer" in lower:
                return name
        raise FileNotFoundError("No se encontró el archivo de preguntas con respuestas para el split TEST.")

    def _find_test_images_zip(self, names: List[str]) -> str:
        for name in names:
            lower = name.lower()
            if "__macosx" in lower or not lower.endswith(".zip"):
                continue
            if "image" in lower and "test" in lower:
                return name
        raise FileNotFoundError("No se encontró el ZIP de imágenes para el split TEST.")

    def _build_image_index_from_zip(self, zf: zipfile.ZipFile, zip_root_prefix: Optional[str]) -> None:
        index = {}
        names = zf.namelist()
        prefix = zip_root_prefix or self.zip_root_prefix

        for name in names:
            if name.endswith("/") or not name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            base = os.path.basename(name)
            stem, _ = os.path.splitext(base)
            keys = [base]
            if stem and stem != base:
                keys.append(stem)

            score = int(bool(self.images_subdir and self.images_subdir.lower() in name.lower()))
            if prefix and name.startswith(prefix):
                score += 1

            for key in keys:
                prev = index.get(key)
                if prev is None or score > prev[0]:
                    index[key] = (score, name)

        self._name_to_path = {k: v[1] for k, v in index.items()}
        if not self._name_to_path:
            print("⚠️  ADVERTENCIA: No se construyó índice de imágenes. Verifica images_subdir y el ZIP.")

    def _finalize_samples(self) -> None:
        self._build_candidates_by_category()
        self._filter_samples_without_candidates()
        self._build_candidates_by_category()

        if self.n_rows != "all":
            self.samples = self.samples[:int(self.n_rows)]
            print(f"📊 Reconstruyendo candidatos después de limitar a {self.n_rows} muestras...")
            self._build_candidates_by_category()
            self._filter_samples_without_candidates()
            self._build_candidates_by_category()

    # ------------------------------------------------------------------
    # Utilidades de categorías y candidatos (mantienen la lógica previa)
    # ------------------------------------------------------------------

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
        print(f"📊 Construyendo candidatos desde {len(self.samples)} muestras (split {self.split_label})...")
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

        print(f"📊 Verificación de categorías (split {self.split_label}):")
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

    def _candidate_image_names(self, image_filename: Optional[str], question_id: Optional[str]) -> List[str]:
        raw_candidates = []
        for raw in (image_filename, question_id):
            if not raw:
                continue
            cleaned = raw.strip()
            if cleaned.upper().startswith("Q") and cleaned[1:].isdigit():
                cleaned = cleaned[1:]
            cleaned = os.path.basename(cleaned)
            if cleaned:
                raw_candidates.append(cleaned)

        expanded = []
        for cand in raw_candidates:
            if cand not in expanded:
                expanded.append(cand)
            stem, ext = os.path.splitext(cand)
            if stem and stem not in expanded:
                expanded.append(stem)
            if not ext:
                for extra in (".jpg", ".jpeg", ".png"):
                    variant = f"{cand}{extra}"
                    if variant not in expanded:
                        expanded.append(variant)
        return expanded

    def _resolve_image_path(self, zf: zipfile.ZipFile, image_filename: Optional[str], question_id: Optional[str]) -> Optional[str]:
        candidates = self._candidate_image_names(image_filename, question_id)

        for cand in candidates:
            path = self._name_to_path.get(cand)
            if path:
                return path

        names = zf.namelist()
        for cand in candidates:
            cand_lower = cand.lower()
            for name in names:
                if name.lower().endswith(cand_lower):
                    return name

        return None

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

        if not hasattr(self, 'candidates_per_cat') or self.candidates_per_cat is None:
            self._build_candidates_by_category()

        if category not in self.candidates_per_cat:
            print(f"⚠️  ADVERTENCIA: Muestra {idx} tiene categoría '{category}' que no existe en candidates_per_cat")
            print(f"   - image_id: {question_id}")
            print(f"   - question: {question[:80]}...")
            print(f"   - answer: {answer}")
            print(f"   Categorías disponibles: {sorted(self.candidates_per_cat.keys())}")
            raise ValueError(
                f"Muestra {idx} (image_id={question_id}) tiene categoría '{category}' no registrada."
            )

        candidates = self.candidates_per_cat.get(category)
        if not candidates:
            raise ValueError(f"Dataset inconsistente: categoría {category} no tiene candidatos.")
        candidates = list(candidates)

        zf = self._open_test_images_zip() if self.split_key == "test" else self._open_primary_zip()

        try:
            image_path = self._resolve_image_path(zf, image_filename, question_id)
            if image_path is None:
                raise KeyError(
                    f"No se pudo encontrar imagen para {question_id} (split {self.split_label}). "
                    f"Revisa la estructura del ZIP y el mapeo pregunta-imagen."
                )

            with zf.open(image_path) as f:
                image = Image.open(BytesIO(f.read())).convert("RGB")
        finally:
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
                "split": self.split_label.lower()
            }
        }
