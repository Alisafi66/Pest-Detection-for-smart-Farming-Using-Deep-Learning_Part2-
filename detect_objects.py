import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import requests
from PIL import Image
from transformers import AutoProcessor, AutoModel

image_path = "/mnt/k/ml-aim/flower.jpg"
image = Image.open(image_path).convert("RGB")
text = ["pest", "flower", "insect", "background", "unknown"]

processor = AutoProcessor.from_pretrained(
    "apple/aimv2-large-patch14-224-lit",
)
model = AutoModel.from_pretrained(
    "apple/aimv2-large-patch14-224-lit",
    trust_remote_code=True,
)

inputs = processor(
    images=image,
    text=text,
    add_special_tokens=True,
    truncation=True,
    padding=True,
    return_tensors="pt",
)
outputs = model(**inputs)
probs = outputs.logits_per_image.softmax(dim=-1)

benchmark_score = 0.895  # AIMv2 ImageNet benchmark score
for label, prob in zip(text, probs[0].tolist()):
    adjusted_confidence = prob * benchmark_score * 100
    print(f"Label: {label}, Confidence: {adjusted_confidence:.2f}%")

# Display the image
image.show()
