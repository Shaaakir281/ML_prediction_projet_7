import pandas as pd
import json
import unittest
import numpy as np

class FlaskApiTest(unittest.TestCase):
    def setUp(self):
        from predict.main import app
        self.app = app.test_client()
        self.app.testing = True
        self.test_data_path = './tests/X_test.csv'  # Assurez-vous que le chemin est correct
        self.y_true_path = './tests/y_test.csv'
        self.test_data = pd.read_csv(self.test_data_path)
        self.y_true = pd.read_csv(self.y_true_path)

    def test_predict_class(self):
        # Envoyer une requête pour tester la prédiction de classe
        json_data = self.test_data.sample(1, random_state=42).to_json(orient='records')
        response = self.app.post('/predict_class', data=json_data, content_type='application/json')
        response_data = response.json
        self.assertIn('predictions', response_data)

    def test_predict_proba_shap(self):
        # Envoyer une requête pour tester les probabilités et valeurs SHAP
        json_data = self.test_data.sample(1, random_state=42).to_json(orient='records')
        response = self.app.post('/predict_proba_shap', data=json_data, content_type='application/json')
        self.assertEqual(response.status_code, 200)
        response_json = response.json
        self.assertIn('probabilities', response_json)
        self.assertIn('shap_values', response_json)
        self.assertIn('base_value', response_json)

    def test_health(self):
        # Tester la route de vérification de l'état de santé
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data.decode(), "API is up and running")

if __name__ == '__main__':
    unittest.main()
