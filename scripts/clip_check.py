from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

image_encoder_path = 'openai/clip-vit-large-patch14'
# image_encoder_path = '/mnt/petrelfs/liuwenran/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41'
# image_encoder_2_path = 'laion/CLIP-ViT-bigG-14-laion2B-39B-b160k'
# image_encoder_2_path ='/mnt/petrelfs/liuwenran/.cache/huggingface/hub/models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k/snapshots/bc7788f151930d91b58474715fdce5524ad9a189'

model = CLIPModel.from_pretrained(image_encoder_path)
processor = CLIPProcessor.from_pretrained(image_encoder_path)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

import ipdb;ipdb.set_trace();

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) 