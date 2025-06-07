from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pandas as pd

app = Flask(__name__)

# Load model ML
model = load_model('best_model.h5', compile=False)

# Label kelas
labels = [
    "bibimbap", "caesar_salad", "chicken_curry", "club_sandwich", "dumplings", "eggs_benedict",
    "falafel", "fried_rice", "grilled_salmon", "hamburger", "lasagna", "miso_soup",
    "omelette", "pho", "ramen", "spaghetti_bolognese", "steak", "sushi", "sashimi",
    "apple_pie", "cheesecake", "chicken_wings", "chocolate_mousse", "churros", "fish_and_chips",
    "french_fries", "fried_calamari", "garlic_bread", "ice_cream", "macaroni_and_cheese",
    "pancakes", "pizza", "red_velvet_cake", "spring_rolls", "tacos", "tiramisu"
]

# Status nutrisi per label
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

# Dictionary untuk menyimpan data gizi
nutrition_dict = {}

def load_nutrition_data():
    global nutrition_dict
    df = pd.read_csv('nutrition_summaryy.csv')
    df.columns = df.columns.str.strip().str.lower()  # normalisasi nama kolom

    # Kolom yang kita perlukan: 'food', 'calories (kcal)', 'fat (g)', 'carbs (g)', 'protein (g)'
    df['food'] = df['food'].str.strip().str.lower()
    nutrition_dict = df.set_index('food').to_dict(orient='index')

load_nutrition_data()

@app.route('/test', methods=['GET'])
def home():
    return "hello world"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    try:
        # Preprocess image
        img = Image.open(file.stream).convert("RGB")
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predict
        prediction = model.predict(img_array)
        if isinstance(prediction, list):
            prediction = prediction[0]
        prediction = np.array(prediction)
        preds_flat = prediction[0].flatten()
        top_3_indices = preds_flat.argsort()[-3:][::-1]

        top_3 = []
        for i in top_3_indices:
            label = labels[i]
            nutrition = nutrition_dict.get(label.lower(), {})
            top_3.append({
                "label": label,
                "confidence": float(preds_flat[i]),
                "nutrition_status": nutrition_labels.get(label, "Tidak diketahui"),
                "nutrition": {
                    "kalori": nutrition.get("calories (kcal)", None),
                    "lemak": nutrition.get("fat (g)", None),
                    "karbohidrat": nutrition.get("carbs (g)", None),
                    "protein": nutrition.get("protein (g)", None)
                }
            })

        return jsonify({'top_predictions': top_3})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
