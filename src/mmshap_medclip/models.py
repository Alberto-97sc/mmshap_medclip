import torch
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


