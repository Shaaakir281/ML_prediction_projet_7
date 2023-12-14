from flask import Flask, request, jsonify
import mlflow.pyfunc

# Initialisation de l'application Flask
app = Flask(__name__)

# Chargement du modèle
model = mlflow.pyfunc.load_model("./artifacts/model_artifact/model.pkl")

# Route pour les prédictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extraction des données de la requête
        data = request.get_json()
        # Prédiction
        predictions = model.predict(data)
        return jsonify(predictions)
    except Exception as e:
        return jsonify({"error": str(e)})

# Route de vérification de l'état
@app.route('/health', methods=['GET'])
def health_check():
    return "API is up and running"

# Démarrage de l'application
if __name__ == '__main__':
    app.run(debug=True)
