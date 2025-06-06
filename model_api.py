from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

app = Flask(__name__)

model = load_model('best_model.h5', compile=False)

# Daftar label kelas sesuai urutan output model
labels = [
    "bibimbap", "caesar_salad", "chicken_curry", "club_sandwich", "dumplings", "eggs_benedict",
    "falafel", "fried_rice", "grilled_salmon", "hamburger", "lasagna", "miso_soup",
    "omelette", "pho", "ramen", "spaghetti_bolognese", "steak", "sushi", "sashimi",
    "apple_pie", "cheesecake", "chicken_wings", "chocolate_mousse", "churros", "fish_and_chips",
    "french_fries", "fried_calamari", "garlic_bread", "ice_cream", "macaroni_and_cheese",
    "pancakes", "pizza", "red_velvet_cake", "spring_rolls", "tacos", "tiramisu"
]

# Status nutrisi tiap label
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

@app.route('/test', methods=['GET'])
def home():
    return "hello world"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    try:
        # Buka dan preprocess gambar
        img = Image.open(file.stream).convert("RGB")
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Prediksi
        prediction = model.predict(img_array)

        # Debug tipe dan isi prediction
        print("type(prediction):", type(prediction))
        print("prediction:", prediction)

        # Jika output model berupa list (multi-output), ambil output pertama
        if isinstance(prediction, list):
            prediction = prediction[0]

        # Pastikan prediction adalah numpy array
        prediction = np.array(prediction)

        print("prediction shape:", prediction.shape)
        print("prediction[0]:", prediction[0])

        # Flatten prediksi untuk urutkan confidence tertinggi
        preds_flat = prediction[0].flatten()

        # Ambil 3 indeks dengan confidence tertinggi
        top_3_indices = preds_flat.argsort()[-3:][::-1]

        # Buat list prediksi top 3
        top_3 = [
            {
                "label": labels[i],
                "confidence": float(preds_flat[i]),
                "nutrition_status": nutrition_labels.get(labels[i], "Tidak diketahui")
            }
            for i in top_3_indices
        ]

        return jsonify({'top_predictions': top_3})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
