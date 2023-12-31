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
        # Chemin vers les fichiers de données de test dans le dépôt
        test_data_path = './tests/X_test.csv'  # Assurez-vous que le chemin est correct
        y_true_path = './tests/y_test.csv'

        # Charger des données de test à partir des fichiers
        test_data = pd.read_csv(test_data_path)
        y_true = pd.read_csv(y_true_path)

        # Convertir le DataFrame en JSON
        json_data = test_data.to_json(orient='records')

        # Envoyer la requête POST
        response = self.app.post('/predict', data=json_data, content_type='application/json')
        self.assertEqual(response.status_code, 200)

        # Vérifier la réponse
        response_data = json.loads(response.data.decode())
        # vérifier si la réponse contient des 1 et des 0 comme prédictions
        print_metrics(y_true, response_data)

if __name__ == '__main__':
    unittest.main()
