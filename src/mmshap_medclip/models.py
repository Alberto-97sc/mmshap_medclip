from types import SimpleNamespace
from typing import Iterable, Sequence

import torch
import open_clip
from transformers import CLIPModel, CLIPProcessor, VisionTextDualEncoderModel, VisionTextDualEncoderProcessor

from mmshap_medclip.registry import register_model

class CLIPWrapper(torch.nn.Module):
    def __init__(self, model, processor):
        super().__init__()
        self.model = model.eval()
        self.processor = processor
        self.tokenizer = processor.tokenizer

    def forward(self, **inputs):
        return self.model(**inputs)


class OpenCLIPTokenizerAdapter:
    """Adapta el tokenizador de open_clip a la interfaz usada en el repo."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self.pad_token_id = getattr(tokenizer, "pad_token_id", 0)
        # open_clip expone sot/eot como enteros
        self.bos_token_id = getattr(tokenizer, "sot_token", None)
        self.eos_token_id = getattr(tokenizer, "eot_token", None)
        self.all_special_tokens = ["<start_of_text>", "<end_of_text>"]

    def __call__(self, texts: Sequence[str], **kwargs):
        return self._tokenizer(texts, **kwargs)

    def convert_ids_to_tokens(self, ids: Iterable[int]):
        decoder = getattr(getattr(self._tokenizer, "tokenizer", None), "decoder", None)
        if decoder is None:
            return [str(int(i)) for i in ids]
        return [decoder.get(int(i), "") for i in ids]

    def __getattr__(self, item):
        return getattr(self._tokenizer, item)


class OpenCLIPProcessor:
    """Processor compatible con prepare_batch para modelos de open_clip."""

    def __init__(self, image_transform, tokenizer: OpenCLIPTokenizerAdapter):
        self.image_transform = image_transform
        self.tokenizer = tokenizer

    def process_images(self, images):
        if not isinstance(images, (list, tuple)):
            images = [images]
        tensor_list = []
        for im in images:
            if hasattr(im, "convert"):
                im = im.convert("RGB")
            tensor_list.append(self.image_transform(im))
        return torch.stack(tensor_list)

    def process_texts(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        token_ids = self.tokenizer(texts)
        if isinstance(token_ids, torch.Tensor):
            input_ids = token_ids
        else:
            input_ids = torch.tensor(token_ids, dtype=torch.long)
        attention_mask = (input_ids != getattr(self.tokenizer, "pad_token_id", 0)).long()
        return input_ids, attention_mask

    def __call__(self, text=None, images=None, return_tensors="pt", padding=True):  # noqa: D401 (firma similar a HF)
        if text is None or images is None:
            raise ValueError("Se requieren 'text' e 'images' para el processor de OpenCLIP.")
        input_ids, attention_mask = self.process_texts(text)
        pixel_values = self.process_images(images)
        n_text = input_ids.shape[0]
        n_img = pixel_values.shape[0]
        if n_img == 1 and n_text > 1:
            pixel_values = pixel_values.expand(n_text, -1, -1, -1).clone()
        elif n_img != n_text:
            raise ValueError(f"Batch desbalanceado: textos={n_text} imágenes={n_img}")
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }


class OpenCLIPWrapper(torch.nn.Module):
    def __init__(self, model, processor, tokenizer_adapter):
        super().__init__()
        self.model = model.eval()
        self.processor = processor
        self.tokenizer = tokenizer_adapter
        visual = getattr(model, "visual", None)

        patch_size = getattr(visual, "patch_size", None)
        if isinstance(patch_size, (list, tuple)):
            patch_tuple = tuple(int(p) for p in patch_size)
        elif patch_size is not None:
            patch_tuple = (int(patch_size), int(patch_size))
        else:
            patch_tuple = None

        image_size = getattr(visual, "image_size", None)
        if isinstance(image_size, (list, tuple)):
            image_size = int(image_size[0])
        elif image_size is not None:
            image_size = int(image_size)
        else:
            proc_image_size = getattr(processor, "image_size", None)
            if isinstance(proc_image_size, (list, tuple)):
                image_size = int(proc_image_size[0])
            else:
                image_size = proc_image_size

        if patch_tuple is None:
            # WhyXrayCLIP usa ViT-L/14 @224 por defecto; usa ese fallback.
            patch_tuple = (14, 14)
        if image_size is None:
            image_size = 224

        self.patch_size = patch_tuple
        self.vision_patch_size = patch_tuple
        self.vision_input_size = image_size

    def forward(self, input_ids=None, pixel_values=None, **_):
        if input_ids is None or pixel_values is None:
            raise ValueError("OpenCLIPWrapper requiere 'input_ids' y 'pixel_values'.")

        # Usar el método manual que funciona correctamente
        image_features = self.model.encode_image(pixel_values)
        text_features = self.model.encode_text(input_ids)

        if hasattr(self.model, "logit_scale"):
            logit_scale = self.model.logit_scale.exp()
        else:
            logit_scale = torch.tensor(1.0, device=image_features.device, dtype=image_features.dtype)

        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return SimpleNamespace(logits_per_image=logits_per_image, logits_per_text=logits_per_text)

@register_model("pubmedclip-vit-b32")
def _mk_pubmedclip(params):
    device = params.get("_device", torch.device("cpu"))
    model = CLIPModel.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32").to(device).eval()
    proc  = CLIPProcessor.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32")
    return CLIPWrapper(model, proc)

@register_model("openai-clip-vit-b32")
def _mk_openai_clip(params):
    device = params.get("_device", torch.device("cpu"))
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return CLIPWrapper(model, proc)


@register_model("whyxrayclip")
def _mk_whyxrayclip(params):
    device = params.get("_device", torch.device("cpu"))
    model_name = params.get("model_name", "hf-hub:yyupenn/whyxrayclip")
    pretrained = params.get("pretrained")

    # El modelo WhyXrayCLIP se publica en Hugging Face y open_clip lo carga
    # usando el identificador completo "hf-hub:<repo_id>" como nombre de
    # modelo. Si el usuario pasa dicho identificador en ``pretrained`` (por
    # compatibilidad con otras configuraciones del repositorio) reutilizamos
    # ese valor como nombre de modelo y evitamos forzar un tag inexistente.
    create_kwargs = {}
    model_name_for_create = model_name
    if (
        pretrained
        and pretrained.startswith("hf-hub:")
        and not model_name.startswith("hf-hub:")
    ):
        model_name_for_create = pretrained
        pretrained = None

    if pretrained:
        create_kwargs["pretrained"] = pretrained

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name_for_create, **create_kwargs
    )

    tokenizer_name = params.get(
        "tokenizer_name",
        model_name_for_create if not model_name_for_create.startswith("hf-hub:") else "ViT-L-14",
    )

    tokenizer = OpenCLIPTokenizerAdapter(open_clip.get_tokenizer(tokenizer_name))
    processor = OpenCLIPProcessor(preprocess, tokenizer)

    wrapper = OpenCLIPWrapper(model.to(device), processor, tokenizer)
    return wrapper

class RclipWrapper(torch.nn.Module):
    """Wrapper para VisionTextDualEncoderModel (Rclip) compatible con el pipeline ISA."""
    
    def __init__(self, model, processor):
        super().__init__()
        self.model = model.eval()
        self.processor = processor
        self.tokenizer = processor.tokenizer
        
        # Extraer patch_size del modelo para SHAP
        vision_config = getattr(model.config, "vision_config", None)
        if vision_config is None:
            vision_config = getattr(model.config, "vision_model_config", None)
        patch_size = None
        if vision_config:
            patch_size = getattr(vision_config, "patch_size", None)
        
        # Si no se encuentra patch_size, usar un valor por defecto razonable
        if patch_size is None:
            patch_size = 32  # Valor por defecto común para modelos ViT
        
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.vision_patch_size = self.patch_size
        
        # Verificar si el modelo tiene logit_scale
        self.logit_scale = getattr(model, "logit_scale", None)
    
    def forward(self, **inputs):
        """Forward que devuelve logits_per_image compatible con el pipeline."""
        # VisionTextDualEncoderModel puede devolver diferentes estructuras
        # Calculamos los logits manualmente para asegurar compatibilidad
        outputs = self.model(**inputs)
        
        # Si ya tiene logits_per_image, usarlo directamente
        if hasattr(outputs, "logits_per_image"):
            return outputs
        
        # Si no, calcular manualmente usando get_image_features y get_text_features
        pixel_values = inputs.get("pixel_values")
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        
        if pixel_values is None or input_ids is None:
            # Intentar usar el forward normal si los métodos no están disponibles
            return outputs
        
        with torch.no_grad():
            img_emb = self.model.get_image_features(pixel_values=pixel_values)
            txt_emb = self.model.get_text_features(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Normalizar embeddings
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
        
        # Calcular similitud coseno
        logits_per_image = img_emb @ txt_emb.T
        
        # Aplicar logit_scale si existe
        if self.logit_scale is not None:
            scale = self.logit_scale.exp() if hasattr(self.logit_scale, "exp") else self.logit_scale
            logits_per_image = scale * logits_per_image
        
        logits_per_text = logits_per_image.T
        
        # Retornar en formato compatible
        return SimpleNamespace(
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text
        )

@register_model("rclip")
def _mk_rclip(params):
    device = params.get("_device", torch.device("cpu"))
    model_name = params.get("model_name", "kaveh/rclip")
    model = VisionTextDualEncoderModel.from_pretrained(model_name).to(device).eval()
    processor = VisionTextDualEncoderProcessor.from_pretrained(model_name)
    
    wrapper = RclipWrapper(model, processor)
    return wrapper
