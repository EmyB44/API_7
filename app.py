import os
import pickle
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import shap
from operator import itemgetter

app = Flask(__name__)

# Chemin du modèle pickle
pickle_model_path = './predict_model_API.pkl'

# Chemin absolu vers le dossier contenant les fichiers numpy
npy_files_dir = './prediction_predict_model_API/'

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

@app.route('/describe', methods=['GET'])
def describe_dataframe():
    # Obtenez le résumé statistique avec la méthode describe()
    describe_df = df.T.describe()

    # Convertissez le DataFrame en un dictionnaire
    describe_dict = describe_df.to_dict()

    return jsonify(describe_dict)



model = loaded_model_pickle.named_steps['model']

# Créez un explainer SHAP basé sur des modèles Tree (XGBoost)
tree_explainer = shap.TreeExplainer(model)

# Dans votre route API
def calculate_feature_importance(model, df):
    # Calculez les SHAP values pour les données données en utilisant l'explainer Tree (XGBoost)
    shap_values = tree_explainer.shap_values(df)[0]
    print (shap_values)

    # Calculez l'importance des fonctionnalités en prenant la moyenne des SHAP values
    feature_importance = {feature: shap_value for feature, shap_value in zip(df.columns, shap_values)}
    print (feature_importance)
    return feature_importance


@app.route('/prediction_feature_importance/<int:id_client>', methods=['GET'])
def prediction_feature_importance(id_client):
    try:
        # Sélectionnez les données du client actuel
        client_data = df[df['SK_ID_CURR'] == id_client]

        if client_data.empty:
            return jsonify({'error': 'Client data not found'}), 404

        # Utilisez votre modèle pour effectuer une prédiction sur ces données
        predictions = loaded_model_pickle.predict(client_data)

        # Calculez les valeurs SHAP pour ces données (basées sur les prédictions)
        shap_values = calculate_feature_importance(predictions, client_data)
        shap_values = {k: float(v) for k, v in shap_values.items()}
        return jsonify(shap_values)

    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)

