import os
import pickle
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import shap

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


@app.route('/id_client/<int:id_client>', methods=['GET'])
def get_client_info(id_client):
    try:
        # Recherchez le client dans le DataFrame en fonction de son ID
        client = df[df['SK_ID_CURR'] == id_client]

        if client.empty:
            return jsonify({'error': 'Client not found'}), 404

        # Convertissez le DataFrame du client en un dictionnaire
        client_info = client.to_dict(orient='records')

        return jsonify(client_info)

    except Exception as e:
        return str(e)




# Définir l'endpoint pour les prédictions
@app.route('/predict/<id_client>', methods=['GET'])
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





@app.route('/predict_proba/<id_client>', methods=['GET'])
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

@app.route('/client_info', methods=['GET'])
def all_client_info():
    try:
        # Convertir l'ensemble du DataFrame des clients en une liste de dictionnaires
        all_clients_info = df.to_dict(orient='records')

        return jsonify(all_clients_info)

    except Exception as e:
        return str(e)



# Définir la fonction de calcul des valeurs SHAP
def calculate_shap_values(client_id, loaded_model_pickle, df, top_n=10):
    # Sélectionnez le client en fonction de l'ID
    client_data = df.loc[df['client_id'] == client_id]

    # Assurez-vous que le client existe
    if client_data.empty:
        return {'error': 'Client not found'}

    # Sélectionnez les fonctionnalités pertinentes pour l'explication SHAP
    X_client = client_data.drop(['client_id'], axis=1)

    # Utilisez le modèle  pour effectuer une prédiction
    prediction = loaded_model_pickle.predict(X_client)

    # Créez un explainer SHAP basé sur le modèle
    explainer = shap.Explainer(model)

    # Calculez les valeurs SHAP pour l'instance du client
    shap_values = explainer.shap_values(X_client)

    # Sélectionnez les N fonctionnalités les plus importantes
    top_n_features_idx = shap_values[0].argsort()[-top_n:][::-1]
    top_n_feature_names = X_client.columns[top_n_features_idx]
    top_n_shap_values = shap_values[0][top_n_features_idx]

    return {'top_features': list(top_n_feature_names), 'shap_values': list(top_n_shap_values)}

# Créer une route pour obtenir les valeurs SHAP d'un client donné
@app.route('/shap_values/<int:client_id>', methods=['GET'])
def get_shap_values(client_id):
    try:
        # Calculer les valeurs SHAP pour le client
        shap_values = calculate_shap_values(client_id, loaded_model_pickle, df)

        # Renvoyer les valeurs SHAP sous forme de JSON
        return jsonify(shap_values)

    except Exception as e:
        return str(e)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)

