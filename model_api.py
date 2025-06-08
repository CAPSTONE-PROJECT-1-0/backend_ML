from flask import Flask, request, jsonify
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import pandas as pd
import os
from flask_cors import CORS

app = Flask(__name__)

# Setup CORS untuk domain frontend & Hapi.js
CORS(app, supports_credentials=True, origins=[
    "https://becapstone-npc01011309-tu16d9a1.leapcell.dev",
    "https://frontend-chi-bice-27.vercel.app"
])

# Load model
model = load_model('best_model.h5', compile=False)

# Labels
labels = [
    "apple_pie", "bibimbap", "caesar_salad", "cheesecake", "chicken_curry", "chicken_wings",
    "chocolate_mousse", "churros", "club_sandwich", "dumplings", "eggs_benedict", "falafel",
    "fish_and_chips", "french_fries", "fried_calamari", "fried_rice", "garlic_bread",
    "grilled_salmon", "hamburger", "ice_cream", "lasagna", "macaroni_and_cheese", "miso_soup",
    "omelette", "pancakes", "pho", "pizza", "ramen", "red_velvet_cake", "sashimi",
    "spaghetti_bolognese", "spring_rolls", "steak", "sushi", "tacos", "tiramisu"
]

# Nutrition status labels
nutrition_labels = {
    "bibimbap": "Seimbang", "caesar_salad": "Seimbang", "chicken_curry": "Seimbang", "club_sandwich": "Seimbang",
    "dumplings": "Seimbang", "eggs_benedict": "Seimbang", "falafel": "Seimbang", "fried_rice": "Seimbang",
    "grilled_salmon": "Seimbang", "hamburger": "Seimbang", "lasagna": "Seimbang", "miso_soup": "Seimbang",
    "omelette": "Seimbang", "pho": "Seimbang", "ramen": "Seimbang", "spaghetti_bolognese": "Seimbang",
    "steak": "Seimbang", "sushi": "Seimbang", "sashimi": "Seimbang",
    "apple_pie": "Tidak_seimbang", "cheesecake": "Tidak_seimbang", "chicken_wings": "Tidak_seimbang",
    "chocolate_mousse": "Tidak_seimbang", "churros": "Tidak_seimbang", "fish_and_chips": "Tidak_seimbang",
    "french_fries": "Tidak_seimbang", "fried_calamari": "Tidak_seimbang", "garlic_bread": "Tidak_seimbang",
    "ice_cream": "Tidak_seimbang", "macaroni_and_cheese": "Tidak_seimbang", "pancakes": "Tidak_seimbang",
    "pizza": "Tidak_seimbang", "red_velvet_cake": "Tidak_seimbang", "spring_rolls": "Tidak_seimbang",
    "tacos": "Tidak_seimbang", "tiramisu": "Tidak_seimbang"
}

# Load nutrition data
nutrition_dict = {}
def load_nutrition_data():
    global nutrition_dict
    df = pd.read_csv('nutrition_summaryy.csv')
    df.columns = df.columns.str.strip().str.lower()
    df['food'] = df['food'].str.strip().str.lower()
    nutrition_dict = df.set_index('food').to_dict(orient='index')
load_nutrition_data()

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    token = request.headers.get('Authorization')
    if not token:
        return jsonify({'error': 'Missing Authorization token'}), 401

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Ambil user info dari form data atau headers
    user_email = request.form.get('userEmail') or request.headers.get('x-user-email')
    user_name = request.form.get('userName') or request.headers.get('x-user-name')

    if not user_email or not user_name:
        return jsonify({
            'error': 'User information required',
            'details': 'Both userEmail and userName must be provided'
        }), 400

    try:
        # Preprocess image
        img = Image.open(file.stream).convert("RGB")
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        prediction = model.predict(img_array)  # Output biasanya shape (1, num_classes)
        # print(prediction)
        # print("DEBUG: prediction shape", prediction.shape)
        print("DEBUG: prediction content", prediction)
        prediction = prediction[0][0]  # Ambil batch pertama

        # Pastikan prediction berbentuk (1, 36)
        # if prediction.ndim == 2 and prediction.shape[1] == len(labels):
        # if prediction.ndim == 2:
        #     prediction = prediction[0]  # Ambil batch pertama
        # elif prediction.ndim == 1 and len(prediction) == len(labels):
        #     # Jika sudah 1D tapi panjang sesuai label, langsung gunakan
        #     pass
        # else:
        #     return jsonify({
        #         'error': f'Prediction size {prediction.size} does not match number of labels {len(labels)}.'
        #     }), 500

        top_index = np.argmax(prediction)
        label = labels[top_index]
        confidence = float(prediction[top_index])
        nutrition = nutrition_dict.get(label.lower(), {})

        result = {
            "label": label,
            "confidence": confidence,
            "nutrition_status": nutrition_labels.get(label, "Tidak diketahui"),
            "nutrition": {
                "kalori": nutrition.get("calories (kcal)"),
                "lemak": nutrition.get("fat (g)"),
                "karbohidrat": nutrition.get("carbs (g)"),
                "protein": nutrition.get("protein (g)")
            }
        }

        # Save image
        image_filename = f"static/uploads/{label}_{np.random.randint(10000)}.jpg"
        img.save(image_filename)
        image_url = f"https://backendml-production-23c3.up.railway.app/{image_filename}"

        # Send to HAPI
        payload = {
            "email": user_email,
            "name": user_name,
            "imageUrl": image_url,
            "analysisResult": jsonify(result),
            "recommendation": jsonify(result["nutrition_status"])
        }

        headers = {
            "Authorization": token,
            "Content-Type": "application/json"
        }

        hapi_url = "https://becapstone-npc01011309-tu16d9a1.leapcell.dev/upload-history"
        response = requests.post(hapi_url, json=payload, headers=headers)

        if response.status_code != 201:
            print("Failed to send to Hapi.js:", response.text)
            print("Payload sent:", payload)

        return jsonify({
            "status": "success",
            "prediction": result,
            "image_url": image_url
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/test', methods=['GET'])
def test():
    return "API OK"

if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    app.run(host="0.0.0.0", port=8080, debug=True)
