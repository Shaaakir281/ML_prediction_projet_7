# Projet 7 de Data Science - Application de Prédiction de Crédit

Ce projet est une application web développée dans le cadre de mon parcours de Data Science. Elle utilise un modèle de machine learning pour prédire la probabilité de remboursement d'un crédit par un client. L'application est construite avec Flask et utilise MLflow pour la gestion et le suivi du modèle de prédiction.

## Fonctionnalités Clés
- **Modèle de Prédiction** : Utilise LightGBM, un algorithme d'apprentissage automatique puissant et efficace, pour la prédiction de crédit.
- **Gestion du Modèle** : Intègre MLflow pour le suivi des expériences et la gestion des modèles.
- **API Web** : Fournit une interface API pour la prédiction de crédit.

## Technologies Utilisées
- Flask pour la création de l'application web.
- LightGBM pour le modèle de machine learning.
- MLflow pour la gestion et le suivi du modèle.
- Docker pour la conteneurisation et le déploiement de l'application.

## Démarrage Rapide

### Exécution Locale
Pour exécuter l'application localement :

```bash
git clone https://github.com/ML_prediction_projet_7.git
cd ML_prediction_projet_7
pip install -r requirements.txt
python main.py
```

### Utilisation de Docker
Pour exécuter l'application à l'aide de Docker :

1. **Construire l'Image Docker** :
   ```bash
   docker build -t prediction_projet7 .
   ```

2. **Lancer le Conteneur** :
   ```bash
   docker run -p 8000:5000 prediction_projet7
   ```

   L'application sera accessible à `http://localhost:8000`.

## Modèle de Prédiction
Le modèle utilise LightGBM, un algorithme d'apprentissage automatique basé sur des arbres de décision, réputé pour sa vitesse d'exécution et son efficacité sur de grands ensembles de données. Ce modèle a été entraîné sur des données historiques de crédit pour prédire la probabilité de défaut de remboursement.

## API Web
L'application expose une API REST pour la prédiction de crédit. Vous pouvez envoyer des données de crédit au format JSON à l'API, et elle renverra la probabilité de remboursement.

### Exemple de Requête
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"feature1": value1, "feature2": value2, ...}'
```

## Contributions
Les contributions à ce projet sont les bienvenues. Veuillez suivre les bonnes pratiques de développement et fournir des tests pour les nouvelles fonctionnalités.