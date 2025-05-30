import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
# Definir la estructura de los datos de entrada
class StudentData(BaseModel):
    features: list  # Lista de valores de entrada
app = FastAPI(title="Student Depression Prediction API")

@app.get("/")
def home():
    return {"message": "¡API Student Depression Prediction está funcionando correctamente!"}
# Cargar el modelo al iniciar el servidor
@app.on_event("startup")
def load_model():
    global model
    model = joblib.load("student_depression_model.pkl")  # Cargar el modelo guardado
@app.post("/predict", tags=["prediction"])
async def predict_depression(data: StudentData):
    try:
        # Convertir los datos en un array de numpy con el formato correcto
        features = np.array(data.features).reshape(1, -1)

        # Hacer la predicción
        prediction = model.predict(features)[0]  # Devuelve 0 o 1
        probability = model.predict_proba(features).tolist()  # Probabilidades
        # Mapear el resultado a una descripción más clara
        result_text = "En depresión" if prediction == 1 else "Sin depresión"
        return {
            "prediction": int(prediction),
            "probabilities": probability,
            "result_text": result_text,
        }
    except Exception as e:
        return {"error": str(e)}

