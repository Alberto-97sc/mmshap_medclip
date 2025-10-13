from types import SimpleNamespace
import torch
from transformers import CLIPModel, CLIPProcessor
from mmshap_medclip.registry import register_model

try:
    import open_clip  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - dependencia opcional en tiempo de import
    open_clip = None

class CLIPWrapper(torch.nn.Module):
    def __init__(self, model, processor):
        super().__init__()
        self.model = model.eval()
        self.processor = processor
        self.tokenizer = processor.tokenizer

    def forward(self, **inputs):
        return self.model(**inputs)


class OpenCLIPTokenizerWrapper:
    """Pequeño wrapper para exponer atributos compatibles con los tokenizers de HF."""

    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        encoder = getattr(tokenizer, "encoder", {})

        def _resolve(attr_name, token_fallback, default=None):
            value = getattr(tokenizer, attr_name, None)
            if isinstance(value, str):
                return encoder.get(value, default)
            if value is not None:
                return value
            if token_fallback is not None and encoder:
                return encoder.get(token_fallback, default)
            return default

        self.pad_token_id = _resolve("pad_token_id", "<pad>", 0)
        self.bos_token_id = _resolve("bos_token_id", "<start_of_text>")
        self.eos_token_id = _resolve("eos_token_id", "<end_of_text>")
        self.context_length = getattr(tokenizer, "context_length", None)

    def __call__(self, texts):
        return self._tokenizer(texts)

    def convert_ids_to_tokens(self, ids):
        decoder = getattr(self._tokenizer, "decoder", None)
        if decoder is None:
            return [str(i) for i in ids]
        return [decoder.get(int(i), str(i)) for i in ids]

    def __getattr__(self, item):
        return getattr(self._tokenizer, item)


class OpenCLIPProcessor:
    """Procesador mínimo para modelos de `open_clip` (tokenizer + preprocess)."""

    def __init__(self, preprocess, tokenizer):
        self.preprocess = preprocess
        self.tokenizer = tokenizer

    def _process_images(self, images):
        if images is None:
            return None
        if not isinstance(images, (list, tuple)):
            images = [images]
        tensors = [self.preprocess(im) for im in images]
        pixel_values = torch.stack(tensors)
        return pixel_values

    def _process_text(self, text):
        if text is None:
            return None
        if isinstance(text, str):
            text = [text]
        input_ids = self.tokenizer(text)
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        return input_ids

    def __call__(self, text=None, images=None, return_tensors="pt", padding=True, **_: dict):  # noqa: D401 - compat API
        inputs = {}
        pix = self._process_images(images)
        if pix is not None:
            inputs["pixel_values"] = pix

        ids = self._process_text(text)
        if ids is not None:
            inputs["input_ids"] = ids
            pad_id = getattr(self.tokenizer, "pad_token_id", None)
            if pad_id is not None:
                inputs["attention_mask"] = (ids != pad_id).long()

        return inputs


class OpenCLIPWrapper(torch.nn.Module):
    def __init__(self, model, processor):
        super().__init__()
        self.model = model.eval()
        self.processor = processor
        self.tokenizer = processor.tokenizer

    def forward(self, **inputs):
        image = inputs.get("pixel_values")
        if image is None:
            image = inputs.get("image")

        text = inputs.get("input_ids")
        if text is None:
            text = inputs.get("text")

        outputs = self.model(image=image, text=text)

        if isinstance(outputs, tuple):
            if len(outputs) < 2:
                raise ValueError("El modelo OpenCLIP no devolvió logits para imagen y texto.")
            logits_per_image, logits_per_text = outputs[0], outputs[1]
        elif isinstance(outputs, dict):
            logits_per_image = outputs.get("logits_per_image")
            logits_per_text = outputs.get("logits_per_text")
        else:
            logits_per_image = getattr(outputs, "logits_per_image", None)
            logits_per_text = getattr(outputs, "logits_per_text", None)

        if logits_per_image is None or logits_per_text is None:
            raise ValueError("No se pudieron extraer los logits del modelo OpenCLIP.")

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


@register_model("whyxrayclip-vit-l-14")
def _mk_whyxrayclip(params):
    if open_clip is None:
        raise ImportError("open-clip-torch no está instalado. Ejecuta `pip install open-clip-torch`." )

    device = params.get("_device", torch.device("cpu"))
    pretrained = params.get("pretrained", "hf-hub:yyupenn/whyxrayclip")
    tokenizer_name = params.get("tokenizer", "ViT-L-14")

    model, _, preprocess = open_clip.create_model_and_transforms(pretrained)
    tokenizer = open_clip.get_tokenizer(tokenizer_name)

    model = model.to(device).eval()
    tokenizer_wrapper = OpenCLIPTokenizerWrapper(tokenizer)
    processor = OpenCLIPProcessor(preprocess, tokenizer_wrapper)

    return OpenCLIPWrapper(model, processor)


