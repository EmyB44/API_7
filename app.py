import os
import pickle
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

app = Flask(__name__)

# Chemin du modèle pickle
pickle_model_path = 'C:/Users/emili/PycharmProjects/flaskProject/predict_model_API.pkl'

# Chemin absolu vers le dossier contenant les fichiers numpy
npy_files_dir = 'C:/Users/emili/PycharmProjects/flaskProject/prediction_predict_model_API/'  # Utilisez le chemin correct

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


# Limiter le nombre de lignes globalement à 10 pour toutes les réponses JSON
@app.before_request
def limit_response_rows():
    max_rows = 1000

    if 'max_rows' in request.args:
        try:
            max_rows = int(request.args['max_rows'])
        except ValueError:
            pass

    global df
    df = df.head(max_rows)

@app.route("/")
def index():
    return "hello world"

# Définir l'endpoint pour les prédictions
@app.route('/predict/<id_client>', methods=['POST'])
def predict(id_client):
    #print(str(id_client))
    client= df.loc[df.SK_ID_CURR==int(id_client)]
    #print(client)
    try:
        # Effectuer des prédictions en utilisant le modèle pickle sur le DataFrame df
        predictions = loaded_model_pickle.predict(client)

        # Renvoyer les prédictions au format JSON
        return jsonify(predictions.tolist())
    except Exception as e:
        return str(e)

@app.route('/predict_proba/<id_client>', methods=['POST'])
def predict_proba(id_client):
    #print(str(id_client))
    client= df.loc[df.SK_ID_CURR==int(id_client)]
    #print(client)
    try:
        # Effectuer des prédictions en utilisant le modèle pickle sur le DataFrame df
        predictions = loaded_model_pickle.predict_proba(client)

        # Renvoyer les prédictions au format JSON
        return jsonify(predictions.tolist())
    except Exception as e:
        return str(e)

# Endpoint pour récupérer les informations sur les clients acceptés, refusés et à évaluer
@app.route('/client_info/<id_client>', methods=['POST'])
def client_info(id_client):
    client = df.loc[df.SK_ID_CURR == int(id_client)]
    try:
        if client.empty:
            return jsonify({'error': 'Client not found'}), 404

        # Convertir le DataFrame du client en un dictionnaire
        client_info = client.to_dict(orient='records')

        return jsonify(client_info)

    except Exception as e:
        return str(e)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
