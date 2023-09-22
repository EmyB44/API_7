import os
import pickle
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

app = Flask(__name__)

# Chemin du modèle pickle
pickle_model_path = 'C:/Users/emili/PycharmProjects/flaskProject/predict_model.pkl'

# Chemin absolu vers le dossier contenant les fichiers numpy
npy_files_dir = 'C:/Users/emili/PycharmProjects/flaskProject/prediction_predict_model/'  # Utilisez le chemin correct

# Emplacement du fichier CSV
csv_file = 'C:/Users/emili/PycharmProjects/flaskProject/data/X_train.csv'

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

# Définir l'endpoint pour les prédictions
@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Effectuer des prédictions en utilisant le modèle pickle sur le DataFrame df
        predictions = loaded_model_pickle.predict(df)

        # Renvoyer les prédictions au format JSON
        return jsonify(predictions.tolist())
    except Exception as e:
        return str(e)

# Endpoint pour récupérer les informations sur les clients acceptés, refusés et à évaluer
@app.route('/client_info', methods=['GET'])
def get_client_info():
    try:
        # Vous pouvez maintenant utiliser les tableaux numpy chargés pour fournir ces informations
        return jsonify({
            'clients_acceptes': loaded_npy_files['clients_acceptes'].tolist(),
            'clients_refuses': loaded_npy_files['clients_refuses'].tolist(),
            'clients_a_evaluer': loaded_npy_files['clients_a_evaluer'].tolist()
        })
    except Exception as e:
        return str(e)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
