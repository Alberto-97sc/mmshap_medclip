from types import SimpleNamespace
from typing import Iterable, Sequence

import torch
import open_clip
from transformers import CLIPModel, CLIPProcessor

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
        self.patch_size = getattr(getattr(model, "visual", None), "patch_size", None)

    def forward(self, input_ids=None, pixel_values=None, **_):
        if input_ids is None or pixel_values is None:
            raise ValueError("OpenCLIPWrapper requiere 'input_ids' y 'pixel_values'.")
        logits_per_image, logits_per_text = self.model(image=pixel_values, text=input_ids)
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


