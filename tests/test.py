import pandas as pd
import json
import unittest
import numpy as np
class FlaskApiTest(unittest.TestCase):
    def setUp(self):
        from predict.main import app
        self.app = app.test_client()
        self.app.testing = True
        self.test_data_path = './tests/X_test.csv'
        self.y_true_path = './tests/y_test.csv'
        self.test_data = pd.read_csv(self.test_data_path)
        self.y_true = pd.read_csv(self.y_true_path)
        
    def test_predict_class(self):
        json_data = self.test_data.sample(1, random_state=42).to_json(orient='records')
        response = self.app.post('/predict_class', data=json_data, content_type='application/json')
        self.assertEqual(response.status_code, 200)
        response_data = response.get_json()
        self.assertIn('predictions', response_data)
    
    def test_predict_proba_shap(self):
        json_data = self.test_data.sample(1, random_state=42).to_json(orient='records')
        response = self.app.post('/predict_proba_shap', data=json_data, content_type='application/json')
        self.assertEqual(response.status_code, 200)
        response_data = response.get_json()
        self.assertIn('probabilities', response_data)
        self.assertIn('shap_values', response_data)
        self.assertIn('base_value', response_data)

    def test_health(self):
        response = self.app.get('/health')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data.decode(), "API is up and running")

if __name__ == '__main__':
    unittest.main()