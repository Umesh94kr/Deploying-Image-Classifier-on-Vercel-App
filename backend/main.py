from fastapi import FastAPI, UploadFile 
from PIL import Image
import torch
import json
import urllib.request
from torchvision import models, transforms
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all (for dev)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
classes = urllib.request.urlopen(url).read().decode("utf-8").splitlines()
model = models.resnet18(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# checking health of API 
@app.get('/')
def home():
    return {'message' : 'API is working'}

@app.post('/predict')
def predict(file: UploadFile):
    image = Image.open(file.file).convert("RGB")
    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        _, pred = torch.max(outputs, 1)
        label = classes[pred.item()]

    return {"prediction": label}
