import os
os.system("pip install tensorflow-addons==0.23.0")

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from model import model_predict

# تعريف FastAPI
app = FastAPI()

# تعريف المدخلات عبر Pydantic
class DrugInteractionRequest(BaseModel):
    drug1: str
    drug2: str

# دالة model_predict المعدلة

    
    

# إنشاء API endpoint
@app.post("/predict/")
async def predict_interaction(request: DrugInteractionRequest):
    drug1 = request.drug1
    drug2 = request.drug2

    # استدعاء دالة التنبؤ
    result = model_predict(drug1, drug2)

    # إرجاع النتيجة
    if result == "Pair not found":
        return {"error": "Pair not found"}
    return {"predicted_class": result}

