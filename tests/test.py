import pandas as pd
import json
from predict.main import app
import unittest
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def print_metrics(y_true, y_pred):
    print("Accuracy: {}".format(accuracy_score(y_true, y_pred)))
    print("Precision: {}".format(precision_score(y_true, y_pred)))
    print("Recall: {}".format(recall_score(y_true, y_pred)))
    print("F1: {}".format(f1_score(y_true, y_pred)))
    print("ROC AUC: {}".format(roc_auc_score(y_true, y_pred)))

class FlaskApiTest(unittest.TestCase):
    # Setup avant chaque test
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    # Test de la route de prédiction
    def test_predict(self):
        test_data_path = './tests/X_test.csv'  # Assurez-vous que le chemin est correct
        y_true_path = './tests/y_test.csv'

        test_data = pd.read_csv(test_data_path)
        y_true = pd.read_csv(y_true_path)

        json_data = test_data.to_json(orient='records')

        response = self.app.post('/predict', data=json_data, content_type='application/json')
        self.assertEqual(response.status_code, 200)

        response_data = json.loads(response.data.decode())

        # S'assurer que la réponse contient les champs attendus
        self.assertIn('predictions', response_data)
        self.assertIn('probabilities', response_data)
        self.assertIn('shap_values', response_data)

        # Extraire les prédictions pour calculer les métriques
        y_pred = response_data['predictions']

        # Afficher les métriques
        print_metrics(y_true, y_pred)

if __name__ == '__main__':
    unittest.main()