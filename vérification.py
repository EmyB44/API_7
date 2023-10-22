import pickle
# ou
import joblib
from xgboost import XGBClassifier


# Utilisation de pickle
with open('./predict_model_API.pkl', 'rb') as file:
    model = pickle.load(file)

# Vérifiez si le modèle est un XGBClassifier
if isinstance(model, XGBClassifier):
    print("Informations sur le modèle XGBoost :")
    print(model.get_params())  # Affiche les paramètres du modèle
else:
    print("Le modèle n'est pas un XGBoostClassifier.")


print(model)
attributes = dir(model)
print(attributes)
print(model.get_params())
