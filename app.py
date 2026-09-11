from flask import Flask, request, render_template, jsonify
import json, joblib
import numpy as np

app = Flask(__name__)

import xgboost as xgb
model = xgb.XGBClassifier()
model.load_model("model.json")
#print(model)

import shap
explainer = shap.TreeExplainer(model)

'''
import numpy as np
x = np.array([[35, 75000, 710]])
prediction = model.predict(x)
shap_values = explainer.shap_values(x)
'''

def preprocess(symptom_labels, user_symptoms):
    user_data = np.zeros(377, dtype=np.float64)
    for us in user_symptoms:
        if us in symptom_labels:
            print(us)
            i = symptom_labels.index(us)
            user_data[i] = 1.0
    return np.array([user_data])

def makePrecition(symptom_labels, user_data, user_input, le):
    probabilities = model.predict_proba(user_data)

    #most likely
    max_prob_index = np.argmax(probabilities, axis=1)
    predicted_label = le.inverse_transform(max_prob_index)[0]
    confidence = np.max(probabilities) * 100


    #maybe switch this so it shows just of the symptoms they DO have
    shap_values = explainer.shap_values(user_data)[0][0]
    print(shap_values)
    contributing_sypmtoms = []
    absolute_data = np.abs(shap_values)
    for x in range(5):
        max_sv_index = np.argmax(absolute_data)
        top_symptom = symptom_labels[max_sv_index]
        absolute_data[max_sv_index] = -np.inf
        explanation = ""
        if top_symptom in user_input:
            explanation = "having " + top_symptom
        else:
            explanation = "not having " + top_symptom
        contributing_sypmtoms.append(explanation)



    #least likely
    min_prob_index = np.argmin(probabilities, axis=1)
    predicted_label3 = le.inverse_transform(min_prob_index)[0]
    confidence3 = np.min(probabilities) * 100

    #second most likely
    probabilities[0][max_prob_index] = -np.inf #get rid of true max
    max_prob_index = np.argmax(probabilities, axis=1)
    predicted_label2 = le.inverse_transform(max_prob_index)[0]
    confidence2 = np.max(probabilities) * 100


    return {
    "Prediction 1": (predicted_label, confidence, contributing_sypmtoms),
    "Prediction 2": (predicted_label2, confidence2),
    "Least Likely": (predicted_label3, confidence3)
    }


@app.route("/", methods=["GET", "POST"])
def home():
    results = None

    with open("symptoms.json") as f:
        symptoms = json.load(f)
    label_encoder = joblib.load("label_encoder.pkl")

    if request.method == "POST":
        user_input = request.form.getlist("checked")
        user_data = preprocess(symptoms, user_input)
        print(user_data)
        results = makePrecition(symptoms, user_data, user_input, label_encoder)


    return render_template("index.html", results=results)


@app.route("/find_care", methods=["GET"])
def find_care():

    # Get user location sent from the frontend browser
    lat = request.args.get('lat')
    lon = request.args.get('lon')
    radius = 5000  # Search distance in meters

    if not lat or not lon:
        return jsonify({"error": "Missing coordinates"}), 400

    # Overpass API query format targeting amenities labeled as "hospital"
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json];
    node["amenity"="hospital"](around:{radius},{lat},{lon});
    out body;
    """
    
    try:
        response = requests.post(overpass_url, data={'data': overpass_query})
        data = response.json()
        
        hospitals = []
        for element in data.get('elements', []):
            hospitals.append({
                "name": element.get('tags', {}).get('name', 'Unnamed Hospital'),
                "lat": element.get('lat'),
                "lon": element.get('lon')
            })
            
        return jsonify(hospitals)
    except Exception as e:
        return jsonify({"error": str(e)}), 500



    return render_template("find_care.html")





if __name__ == "__main__":
    app.run(debug = True)