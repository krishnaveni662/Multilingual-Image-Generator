from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from io import BytesIO
import requests

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

prompt = "a cat sitting on a couch"
image = Image.open(BytesIO(requests.get("https://img.freepik.com/free-photo/close-up-beautiful-cat_23-2149216352.jpg").content))
inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
outputs = model(**inputs)
clip_score = outputs.logits_per_image.item()
print(f"CLIPScore: {clip_score:.3f}")




from pycocoevalcap.cider.cider import Cider

# Example references and predictions
references = {
    0: ["a small cat sitting on a sofa"],
    1: ["a man riding a bike"],
    2: ["a group of people standing in the park"]
}
predictions = {
    0: ["a cat sitting on a couch"],
    1: ["a person riding a motorcycle"],
    2: ["people gathered in a garden"]
}

cider_scorer = Cider()
score, scores = cider_scorer.compute_score(references, predictions)
print(f"\nAverage CIDEr Score: {score:.3f}")


from jiwer import wer, cer

# Reference = ground truth spoken text
reference = "Hello everyone welcome to the AI project"

# Prediction = output from your speech recognition model
prediction = "Hello every one welcome to AI project"

# Calculate metrics
wer_score = wer(reference, prediction)
cer_score = cer(reference, prediction)

print(f"WER: {wer_score:.3f}")
print(f"CER: {cer_score:.3f}")
