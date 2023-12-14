import pandas as pd
import json
from ML_prediction_projet_7.predict.main import app
import unittest

class FlaskApiTest(unittest.TestCase):
    # Setup avant chaque test
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    # Test de la route de prédiction
    def test_predict(self):
        # Charger des données de test à partir d'un fichier CSV
        test_data = pd.read_csv('./sample_test_data.csv')
        
        # Convertir le DataFrame en JSON
        json_data = test_data.to_json(orient='records')

        # Envoyer la requête POST
        response = self.app.post('/predict', data=json_data, content_type='application/json')
        self.assertEqual(response.status_code, 200)

        # Vérifier la réponse
        response_data = json.loads(response.data.decode())
        # vérifier si la réponse contient des 1 et des 0 comme prédictions
        for prediction in response_data:
            self.assertIn(prediction, [0, 1])

if __name__ == '__main__':
    unittest.main()
