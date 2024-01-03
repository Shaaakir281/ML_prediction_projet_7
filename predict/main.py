import os
from flask import Flask, request, jsonify
import mlflow.pyfunc
import mlflow.sklearn
import pandas as pd
import shap
import numpy as np

app = Flask(__name__)

# Chemin relatif au modèle MLflow
relative_model_path = "model_with_threshold"

# Chargement des modèles
model = mlflow.pyfunc.load_model(relative_model_path)
model_sans_threshold = mlflow.sklearn.load_model("artifacts/model_artifact")

# Chargement de l'explainer SHAP
explainer = shap.TreeExplainer(model_sans_threshold)

@app.route('/predict_class', methods=['POST'])
def predict_class():
    try:
        json_data = request.get_json()
        data = pd.DataFrame(json_data)
        predictions = model.predict(data)
        return jsonify(predictions.tolist())
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/predict_proba_shap', methods=['POST'])
def predict_proba_shap():
    try:
        json_data = request.get_json()
        data = pd.DataFrame(json_data)
        predicted_proba = model_sans_threshold.predict_proba(data)[:, 1]
        shap_values = explainer.shap_values(data)

        # Debug: Affichez les valeurs SHAP avant la conversion
        print("Valeurs SHAP avant conversion:", shap_values)

        # Conversion des valeurs SHAP
        shap_values_json = shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values

        # Debug: Affichez les valeurs SHAP après la conversion
        print("Valeurs SHAP après conversion:", shap_values_json)

        response = {
            "probabilities": predicted_proba.tolist(),
            "shap_values": shap_values_json,
            "base_value": explainer.expected_value.tolist()
        }
        return jsonify(response)
    except Exception as e:
        print("Erreur:", e)
        return jsonify({"error": str(e)})



@app.route('/health', methods=['GET'])
def health_check():
    return "API is up and running"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
