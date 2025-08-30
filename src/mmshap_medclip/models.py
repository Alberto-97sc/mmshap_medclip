import torch
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

class RClipWrapper(torch.nn.Module):
    """Wrapper para RClip (VisionTextDualEncoderModel) que mantiene compatibilidad con el pipeline existente."""
    def __init__(self, model, processor):
        super().__init__()
        self.model = model.eval()
        self.processor = processor
        self.tokenizer = processor.tokenizer
        
    def forward(self, **inputs):
        return self.model(**inputs)
    
    def get_text_features(self, **inputs):
        """Compatibilidad con métodos CLIP estándar."""
        return self.model.get_text_features(**inputs)
    
    def get_image_features(self, **inputs):
        """Compatibilidad con métodos CLIP estándar."""
        return self.model.get_image_features(**inputs)

@register_model("rclip")
def _mk_rclip(params):
    device = params.get("_device", torch.device("cpu"))
    model = VisionTextDualEncoderModel.from_pretrained("kaveh/rclip").to(device).eval()
    proc  = VisionTextDualEncoderProcessor.from_pretrained("kaveh/rclip")
    return RClipWrapper(model, proc)


