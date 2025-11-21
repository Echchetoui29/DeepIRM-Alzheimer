from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import os

app = FastAPI(title="Détection Alzheimer - AlexNet")

# 🔥 CORRECTION : Chemin absolu pour les templates
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

print(f"📁 Dossier templates: {TEMPLATES_DIR}")
print(f"📁 Fichiers dans templates: {os.listdir(TEMPLATES_DIR) if os.path.exists(TEMPLATES_DIR) else 'Dossier inexistant'}")

templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Charger le modèle
MODEL_PATH = os.path.join(BASE_DIR, "models", "alexnet_alzheimer_final.keras")

print(f"📁 Recherche du modèle à: {MODEL_PATH}")
print(f"📁 Le fichier existe: {os.path.exists(MODEL_PATH)}")

try:
    model = load_model(MODEL_PATH)
    print("✅ Modèle AlexNet chargé avec succès")
except Exception as e:
    print(f"❌ Erreur chargement modèle: {e}")
    model = None

# Classes de prédiction
CLASSES = {
    0: "Non Dément",
    1: "Dément Très Léger", 
    2: "Dément Léger",
    3: "Dément Modéré"
}

def preprocess_image(image: Image.Image):
    """Prétraite l'image pour AlexNet"""
    image = image.resize((227, 227))
    image_array = np.array(image).astype('float32') / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
# Créez une route de diagnostic


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    
    if model is None:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "Modèle non disponible. Contactez l'administrateur."
        })
    
    if not file.content_type.startswith('image/'):
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": "Veuillez uploader une image (JPG, PNG, etc.)"
        })
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        predicted_class = CLASSES.get(predicted_class_idx, "Inconnu")
        
        result = {
            "classe": predicted_class,
            "confiance": f"{confidence * 100:.2f}%",
            "filename": file.filename
        }
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": result
        })
        
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "error": f"Erreur de traitement: {str(e)}"
        })
@app.get("/model-info")
async def model_info():
    if model is None:
        return {"error": "Modèle non chargé"}
    
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    
    return {
        "input_shape": str(model.input_shape),
        "output_shape": str(model.output_shape),
        "output_units": model.layers[-1].units if hasattr(model.layers[-1], 'units') else None,
        "summary": "\n".join(model_summary)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)