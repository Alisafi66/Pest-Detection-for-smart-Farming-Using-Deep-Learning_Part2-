import requests
from PIL import Image
from transformers import AutoProcessor, AutoModel

image_path = "/mnt/k/ml-aim/Plant_Pest-Header.jpg" 
image = Image.open(image_path).convert("RGB")
text = ["pest"]

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
from PIL import ImageDraw, ImageFont

# Simulated visualization of detection for saving the image
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()
box = [50, 50, 200, 200]  # Example bounding box
label = f"Pest: {round(probs[0][0].item() * 100, 2)}%"  # Use the probability from the model
draw.rectangle(box, outline="red", width=3)
draw.text((box[0], box[1] - 10), label, fill="red", font=font)

# Save the annotated image
output_image_path = "/mnt/k/ml-aim/detected_image.jpg"
image.save(output_image_path)
print(f"Annotated image saved to: {output_image_path}")
