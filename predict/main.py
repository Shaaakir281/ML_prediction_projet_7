from flask import Flask, request, jsonify
import mlflow.pyfunc
import pandas as pd

# Initialisation de l'application Flask
app = Flask(__name__)

# Chargement du modèle
model = mlflow.pyfunc.load_model("./artifacts/model_artifact/model.pkl")

# Route pour les prédictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extraction des données de la requête en JSON
        json_data = request.get_json()

        # Convertir les données JSON en DataFrame pandas
        if isinstance(json_data, dict):  # Vérifier si les données sont un dictionnaire
            data = pd.DataFrame([json_data])
        else:  # Supposer que les données sont une liste de dictionnaires
            data = pd.DataFrame(json_data)

        # Prédiction
        predictions = model.predict(data)
        return jsonify(predictions.tolist())  # Convertir les prédictions en liste pour la réponse JSON
    except Exception as e:
        return jsonify({"error": str(e)})

# Route de vérification de l'état
@app.route('/health', methods=['GET'])
def health_check():
    return "API is up and running"

# Démarrage de l'application
if __name__ == '__main__':
    app.run(debug=True)
