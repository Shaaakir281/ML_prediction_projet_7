# Utiliser une image de base Python
FROM python:3.9.18

# Mettre à jour et installer les dépendances nécessaires
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail dans le conteneur
WORKDIR /ML_prediction_projet_7

# Copier les fichiers du projet dans le conteneur
COPY . /ML_prediction_projet_7

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port sur lequel l'application va tourner
EXPOSE 5000

# Définir la commande pour démarrer l'application
CMD ["python", "predict/main.py"]