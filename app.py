from flask import Flask, request, jsonify
import joblib
import pandas as pd  # N'oubliez pas d'importer pandas pour gérer le fichier CSV
import mlflow
app = Flask(__name__)



# Charger le modèle MLflow
model_path = 'file:///C:/Users/emili/PycharmProjects/flaskProject/mon_modele'


# Charger le modèle MLflow depuis le répertoire
loaded_model = mlflow.pyfunc.load_model(model_path)




# Emplacement du fichier CSV
csv_file = 'C:\\Users\\emili\\PycharmProjects\\flaskProject\\data\\application_test.csv'

# Chargez les données CSV dans un DataFrame
df = pd.read_csv(csv_file)

# Définir l'endpoint pour les prédictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données d'entrée depuis la requête POST
        data = request.get_json()
        input_data = pd.DataFrame(data)  #

        # Effectuer des prédictions
        predictions = model.predict(input_data)

        # Renvoyer les prédictions au format JSON
        return jsonify(predictions.tolist())
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


