from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import pandas as pd
import os

model = load_model('best_model.h5', compile=False)

labels = [
"bibimbap", "caesar_salad", "chicken_curry", "club_sandwich", "dumplings", "eggs_benedict",
"falafel", "fried_rice", "grilled_salmon", "hamburger", "lasagna", "miso_soup",
"omelette", "pho", "ramen", "spaghetti_bolognese", "steak", "sushi", "sashimi",
"apple_pie", "cheesecake", "chicken_wings", "chocolate_mousse", "churros", "fish_and_chips",
"french_fries", "fried_calamari", "garlic_bread", "ice_cream", "macaroni_and_cheese",
"pancakes", "pizza", "red_velvet_cake", "spring_rolls", "tacos", "tiramisu"
]

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

def load_nutrition_data():
    df = pd.read_csv('nutrition_summaryy.csv')
    df.columns = df.columns.str.strip().str.lower()
    df['food'] = df['food'].str.strip().str.lower()
    return df.set_index('food').to_dict(orient='index')

nutrition_dict = load_nutrition_data()

def predict_from_file(image_path):
    try:
        filename = os.path.basename(image_path)
        img = load_img(image_path, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        prediction = model.predict(img_array)
        if isinstance(prediction, list):
            prediction = prediction[0]

        preds_flat = prediction[0]
        top_3_indices = preds_flat.argsort()[-3:][::-1]

        top_3 = []
        for i in top_3_indices:
            label = labels[i]
            nutrition = nutrition_dict.get(label.lower(), {})
            top_3.append({
                "filename": filename,
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
        return top_3

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return []

if __name__ == '__main__':
    image_path = input("Masukkan relative path gambar (contoh: images/salad1.jpg):\n> ")

    if not os.path.exists(image_path):
        print("❌ File tidak ditemukan. Pastikan path benar.")
    else:
        results = predict_from_file(image_path)
        for item in results:
            print("\n=== Prediction ===")
            print("File:", item['filename'])
            print("Label:", item['label'])
            print("Confidence:", round(item['confidence'], 4))
            print("Status:", item['nutrition_status'])
            print("Nutrition:", item['nutrition'])