import os
import pickle
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

app = Flask(__name__)

# Chemin du modèle pickle
pickle_model_path = './predict_model_API.pkl'

# Chemin absolu vers le dossier contenant les fichiers numpy
npy_files_dir = './prediction_predict_model_API/'  # Utilisez le chemin correct

# Emplacement du fichier CSV
csv_file = './X_train.csv'

# Charger le modèle à partir du fichier pickle
with open(pickle_model_path, 'rb') as file:
    loaded_model_pickle = pickle.load(file)

# Charger tous les fichiers numpy du dossier
loaded_npy_files = {}
for npy_file in os.listdir(npy_files_dir):
    if npy_file.endswith(".npy"):
        file_name = os.path.splitext(npy_file)[0]
        file_path = os.path.join(npy_files_dir, npy_file)
        loaded_npy_files[file_name] = np.load(file_path)

# Charger les données CSV dans un DataFrame
df = pd.read_csv(csv_file)




@app.route("/")
def index():
    return "hello world"

