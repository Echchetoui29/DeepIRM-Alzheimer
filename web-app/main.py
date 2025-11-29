from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import os
import sys
from pathlib import Path
import logging

BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models import AlexNet

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Détection Alzheimer - AlexNet",
    description="API basée sur AlexNet pour la classification MRI liée à la maladie d’Alzheimer",
    version="2.0"
)

TEMPLATES_DIR = BASE_DIR / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

MODEL_PATH = BASE_DIR / "models" / "alexnet_finetuned_acc_97.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_MEAN = 0.2945542335510254
DATASET_STD = 0.3180045485496521

CLASSES = {
    0: "Non Demented",
    1: "Very Mild Demented",
    2: "Mild Demented",
    3: "Moderate Demented"
}

CLASS_DESCRIPTIONS = {
    0: "Aucun signe de démence détecté",
    1: "Début léger de déclin cognitif",
    2: "Déclin cognitif modéré observé",
    3: "Stade avancé avec altération significative"
}

model = None
model_info = {}


def load_model():
    """Charge le modèle AlexNet entraîné et prépare l'inférence."""
    global model, model_info
    try:
        if not MODEL_PATH.exists():
            return False
        
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict):
            model_info['epoch'] = checkpoint.get('epoch', 'unknown')
            model_info['val_accuracy'] = checkpoint.get('val_accuracy', 'unknown')
            model_info['val_loss'] = checkpoint.get('val_loss', 'unknown')

        model = AlexNet(
            num_classes=4,
            input_channels=1,
            dropout_rate=0.498070764044508
        )
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        else:
            model = checkpoint
        
        model.to(device)
        model.eval()
        return True
        
    except Exception:
        return False


def get_transform():
    """Retourne la transformation d’image utilisée pour l’inférence."""
    return transforms.Compose([
        transforms.Resize((200, 190)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[DATASET_MEAN], std=[DATASET_STD])
    ])


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Prétraite une image brute en tenseur prêt pour l’inférence."""
    image = image.convert('L')
    transform = get_transform()
    tensor = transform(image)
    return tensor.unsqueeze(0)


@app.on_event("startup")
async def startup_event():
    """Événement de démarrage du serveur : charge le modèle."""
    load_model()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Affiche l’interface web principale."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "model_loaded": model is not None,
        "model_info": model_info
    })


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    """Effectue une prédiction sur une image envoyée par l'utilisateur."""
    if model is None:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "Modèle indisponible.",
            "model_loaded": False
        })
    
    if not file.content_type or not file.content_type.startswith('image/'):
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "Veuillez envoyer un fichier image valide.",
            "model_loaded": True,
            "model_info": model_info
        })
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        original_size = image.size
        original_mode = image.mode
        
        image_tensor = preprocess_image(image).to(device)
        
        with torch.no_grad():
            logits = model(image_tensor)
            probabilities = torch.softmax(logits, dim=1)
        
        predicted_idx = torch.argmax(probabilities[0]).item()
        confidence = probabilities[0][predicted_idx].item()
        
        all_probs = {
            CLASSES[i]: f"{probabilities[0][i].item() * 100:.2f}%"
            for i in range(len(CLASSES))
        }
        
        result = {
            "class": CLASSES[predicted_idx],
            "class_idx": predicted_idx,
            "confidence": f"{confidence * 100:.2f}%",
            "confidence_value": confidence,
            "description": CLASS_DESCRIPTIONS[predicted_idx],
            "all_probabilities": all_probs,
            "filename": file.filename,
            "image_info": {
                "size": f"{original_size[0]}x{original_size[1]}",
                "mode": original_mode
            }
        }
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": result,
            "model_loaded": True,
            "model_info": model_info
        })
        
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Erreur lors du traitement de l'image : {str(e)}",
            "model_loaded": True,
            "model_info": model_info
        })


@app.get("/api/predict")
async def api_predict(file: UploadFile = File(...)):
    """API JSON pour effectuer une prédiction à partir d'une image."""
    if model is None:
        return JSONResponse(status_code=503, content={"error": "Modèle non disponible"})
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        image_tensor = preprocess_image(image).to(device)
        
        with torch.no_grad():
            logits = model(image_tensor)
            probabilities = torch.softmax(logits, dim=1)
        
        predicted_idx = torch.argmax(probabilities[0]).item()
        confidence = probabilities[0][predicted_idx].item()
        
        all_probs = {CLASSES[i]: float(probabilities[0][i].item()) for i in range(4)}
        
        return JSONResponse(content={
            "prediction": CLASSES[predicted_idx],
            "class_index": predicted_idx,
            "confidence": confidence,
            "all_probabilities": all_probs,
            "description": CLASS_DESCRIPTIONS[predicted_idx]
        })
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/health")
async def health_check():
    """Vérifie l'état du service."""
    return JSONResponse(content={
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    })


@app.get("/model-info")
async def get_model_info():
    """Retourne les informations du modèle chargé."""
    if model is None:
        return JSONResponse(status_code=503, content={"error": "Modèle non chargé"})
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return JSONResponse(content={
        "model_name": "AlexNet (Custom)",
        "architecture": "AlexNet ajusté pour images MRI en niveaux de gris",
        "input_size": [1, 200, 190],
        "num_classes": 4,
        "classes": CLASSES,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "device": str(device),
        "training_info": model_info,
        "normalization": {"mean": DATASET_MEAN, "std": DATASET_STD}
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
