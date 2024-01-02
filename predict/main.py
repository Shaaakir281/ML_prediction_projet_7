import os
from flask import Flask, request, jsonify
import mlflow.pyfunc
import mlflow.sklearn
import pandas as pd
import shap

# Initialisation de l'application Flask
app = Flask(__name__)
# Chemin relatif au dossier contenant le modèle MLflow
relative_model_path = "model_with_threshold"
# Charger le modèle

model = mlflow.pyfunc.load_model(relative_model_path)
model_sans_threshold = mlflow.sklearn.load_model("artifacts/model_artifact")

# Charger l'explicateur SHAP (assurez-vous que le modèle est compatible avec SHAP)
explainer = shap.TreeExplainer(model_sans_threshold)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_data = request.get_json()
        if isinstance(json_data, dict):
            data = pd.DataFrame([json_data])
        elif isinstance(json_data, list):
            data = pd.DataFrame(json_data)
        else:
            data = pd.read_json(json_data, orient='records')

        # Prédiction et probabilités
        predictions = model.predict(data)
        predicted_proba = model_sans_threshold.predict_proba(data)[:, 1]

        # Calcul des valeurs SHAP pour les données reçues
        shap_values = explainer.shap_values(data)

        # Construire la réponse JSON
        response = {
            "predictions": predictions.tolist(),
            "probabilities": predicted_proba.tolist(),
            "shap_values": shap_values
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/health', methods=['GET'])
def health_check():
    return "API is up and running"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
