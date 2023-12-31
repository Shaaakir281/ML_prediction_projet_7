import os
from flask import Flask, request, jsonify
import mlflow.pyfunc
import pandas as pd

# Initialisation de l'application Flask
app = Flask(__name__)

# Utiliser la variable d'environnement pour le chemin du modèle
model_path = os.getenv("MODEL_PATH", os.path.join(os.path.dirname(__file__), "model_with_threshold"))
model = mlflow.pyfunc.load_model(model_path)

# Route pour les prédictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extraction des données de la requête en JSON
        json_data = request.get_json()

        # Convertir les données JSON en DataFrame pandas
        if isinstance(json_data, dict):  # Vérifier si les données sont un dictionnaire
            data = pd.DataFrame([json_data])
        elif isinstance(json_data, list):
            data = pd.DataFrame(json_data)
        else:
            data = pd.read_json(json_data, orient='records')

        # Prédiction
        predictions = model.predict(data)
        return jsonify(predictions.tolist())  # Convertir les prédictions en liste pour la réponse JSON
    except Exception as e:
        # Gérer l'exception et renvoyer un message d'erreur
        return jsonify({"error": str(e)})

# Route de vérification de l'état
@app.route('/health', methods=['GET'])
def health_check():
    return "API is up and running"

# Démarrage de l'application
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
