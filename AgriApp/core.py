import os
import tensorflow as tf
import numpy as np
from django.conf import settings

def model_prediction(test_image_path):
    model_path = os.path.join(settings.BASE_DIR,'AgriApp', 'models', 'trained_model.keras')
    model = tf.keras.models.load_model(model_path)
    image = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image into batch
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

def get_recommendations(disease_name):
    recommendations = {
       'Apple___Apple_scab': {
            'definition': 'This is an apple leaf with "Apple Scab" disease',
            'pesticide': 'Use fungicides like Captan, Carbendazim, or Mancozeb.',
            'fertilizer': 'Cow manure or vermicompost.'
        },
        'Apple___Black_rot': {
            'definition': 'This is an apple leaf with "Black Rot" disease',
            'pesticide': 'Use fungicides such as Mancozeb or Copper oxychloride',
            'fertilizer': 'Compost or farmyard manure (FYM).'
        },
        'Apple___Cedar_apple_rust': {
            'definition': 'This is an apple leaf with "Cedar Apple Rust" disease',
            'pesticide': 'Use fungicides containing Myclobutanil or Copper oxychloride.',
            'fertilizer': ' Neem cake or compost.'
        },
        'Apple___healthy': {
             'definition': 'This is a healthy apple leaf ',
            'pesticide': 'No pesticide recommended as it is stated Healthy',
            'fertilizer': 'Organic fertilizers like compost, vermicompost, and cow manure'
        },
        'Bean___angular_leaf_spot': {
            'definition': 'This is a bean leaf with "Angular Leaf Spot" disease',
            'pesticide': 'Copper-based fungicides like Copper oxychloride',
            'fertilizer': 'Neem cake or compost.'
        },
        'Bean___rust': {
            'definition': 'This is a bean leaf with "Rust" disease',
            'pesticide': 'Use fungicides containing Mancozeb or Chlorothalonil',
            'fertilizer': 'Farmyard manure or vermicompost.'
        },
        'Beans___healthy': {
            'definition': 'This is a healthy bean leaf',
            'pesticide': 'No pesticide recommended as it is stated Healthy',
            'fertilizer': 'Organic fertilizers like compost or vermicompost.'
        },
        'Blueberry___healthy': {
            'definition': 'This is a healthy blueberry leaf',
            'pesticide': 'No pesticide recommended as it is stated Healthy',
            'fertilizer': 'Use organic fertilizers like compost, pine needle mulch, and well-rotted manure.'
        },
        'Cherry_(including_sour)___healthy': {
            'definition': 'This is a healthy cherry leaf',
            'pesticide': 'No pesticide recommended as it is stated Healthy',
            'fertilizer': 'Organic fertilizers like compost or well-rotted manure'
        },
        'Cherry_(including_sour)___Powdery_mildew': {
            'definition': 'This is a cherry leaf with "Powdery Mildew" disease',
            'pesticide': 'Use sulfur-based fungicides or fungicides containing Myclobutanil',
            'fertilizer': 'Vermicompost or compost.'
        },
        'Corn___Common_Rust': {
            'definition': 'This is a corn leaf with "Common Rust" disease',
            'pesticide': 'Use fungicides like Mancozeb or Copper oxychloride',
            'fertilizer': 'Vermicompost or green manure..'
        },
        'Corn___Gray_Leaf_Spot': {
            'definition': 'This is a corn leaf with "Gray Leaf Spot" disease',
            'pesticide': 'Use fungicides like Azoxystrobin or Propiconazole',
            'fertilizer': 'Farmyard manure or compost.'
        },
        'Corn___Healthy': {
            'definition': 'This is a healthy corn leaf',
            'pesticide': 'No pesticide recommended as it is stated Healthy',
            'fertilizer': 'Organic fertilizers like compost or vermicompost.'
        },
        'Corn___Leaf_Blight': {
            'definition': 'This is a corn leaf with "Leaf Blight" disease',
            'pesticide': 'Use fungicides like Azoxystrobin or Propiconazole',
            'fertilizer': 'Farmyard manure or compost.'
        },
        'Grape___Black_rot': {
            'definition': 'This is a grape leaf with "Black Rot" disease',
            'pesticide': 'Use fungicides like Myclobutanil or Mancozeb.',
            'fertilizer': 'Neem cake or compost.'
        },
        'Grape___Esca_(Black_Measles)': {
            'definition': 'This is a grape leaf with "Black Measles" disease',
            'pesticide': 'Use systemic fungicides such as Myclobutanil.',
            'fertilizer': 'Vermicompost or compost.'
        },
        'Grape___healthy': {
            'definition': 'This is a healthy grape leaf',
            'pesticide': 'No pesticide recommended as it is stated Healthy',
            'fertilizer': 'Organic fertilizers like compost or vermicompost.'
        },
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
            'definition': 'This is a grape leaf with "Leaf Blight" disease',
            'pesticide': 'Use fungicides such as Mancozeb or Captan.',
            'fertilizer': 'Farmyard manure or compost.'
        },
        'Invalid': {
            'definition': '',
            'pesticide': '',
            'fertilizer': ''
        },
        'Orange___Haunglongbing_(Citrus_greening)': {
            'definition': 'This is an orange leaf with "Citrus_greening" disease',
            'pesticide': 'Manage with insecticides like Imidacloprid for the Asian citrus psyllid and nutritional sprays with micronutrients. ',
            'fertilizer': 'Neem cake or compost.'
        },
        'Peach___Bacterial_spot': {
            'definition': 'This is a peach leaf with "Bacterial Spot" disease',
            'pesticide': 'Use copper-based bactericides.',
            'fertilizer': 'Compost or farmyard manure.'
        },
        'Peach___healthy': {
            'definition': 'This is a healthy peach leaf',
            'pesticide': 'No pesticide recommended as it is stated Healthy',
            'fertilizer': 'Organic fertilizers like compost or well-rotted manure.'
        },
        'Pepper,_bell___Bacterial_spot': {
            'definition': 'This is a pepper leaf with "Bacterial Spot" disease',
            'pesticide': 'Use copper-based bactericides.',
            'fertilizer': 'Neem cake or vermicompost.'
        },
        'Pepper,_bell___healthy': {
            'definition': 'This is a healthy pepper leaf',
            'pesticide': 'No pesticide recommended as it is stated Healthy',
            'fertilizer': 'Organic fertilizers like compost or vermicompost.'
        },
        'Potato___Early_Blight': {
            'definition': 'This is a potato leaf with "Early Blight" disease',
            'pesticide': 'Use fungicides such as Chlorothalonil or Mancozeb.',
            'fertilizer': 'Vermicompost or farmyard manure.'
        },
        'Potato___Healthy': {
            'definition': 'This is a healthy potato leaf',
            'pesticide': 'No pesticide recommended as it is stated Healthy',
            'fertilizer': 'Organic fertilizers like compost or vermicompost.'
        },
        'Potato___Late_Blight': {
            'definition': 'This is a potato leaf with "Late Blight" disease',
            'pesticide': 'Use fungicides like Chlorothalonil or Mancozeb.',
            'fertilizer': 'Compost or green manure.'
        },
        'Raspberry___healthy': {
            'definition': 'This is a healthy raspberry leaf',
            'pesticide': 'No pesticide recommended as it is stated Healthy',
            'fertilizer': 'Organic fertilizers like compost or well-rotted manure.'
        },
        'Rice___Brown_Spot': {
            'definition': 'This is a rice leaf with "Brown Spot" disease',
            'pesticide': 'Use fungicides like Propiconazole. Organic fertilizer',
            'fertilizer': 'Neem cake or green manure..'
        },
        'Rice___Healthy': {
            'definition': 'This is a healthy rice leaf',
            'pesticide': 'No pesticide recommended as it is stated Healthy',
            'fertilizer': 'Organic fertilizers like compost or vermicompost.'
        },
        'Rice___Hispa': {
            'definition': 'This is a rice leaf with "Hispa" disease',
            'pesticide': 'Use insecticides like Carbaryl.',
            'fertilizer': 'Farmyard manure or vermicompost.'
        },
        'Rice___Leaf_Blast': {
            'definition': 'This is a rice leaf with "Leaf Blast" disease',
            'pesticide': 'Use fungicides containing Tricyclazole or Isoprothiolane.',
            'fertilizer': 'Green manure or compost.'
        },
        'Soybean___healthy': {
            'definition': 'This is a healthy soybean leaf',
            'pesticide': 'No pesticide recommended as it is stated Healthy',
            'fertilizer': 'Organic fertilizers like compost or vermicompost.'
        },
        'Squash___Powdery_mildew': {
            'definition': 'This is a squash leaf with "Powdery Mildew" disease',
            'pesticide': ' Use sulfur-based fungicides or fungicides containing Myclobutanil.',
            'fertilizer': 'Compost or neem cake.'
        },
        'Strawberry___healthy': {
            'definition': 'This is a healthy strawberry leaf',
            'pesticide': 'No pesticide recommended as it is stated Healthy',
            'fertilizer': 'Organic fertilizers like compost or vermicompost.'
        },
        'Strawberry___Leaf_scorch': {
            'definition': 'This is a strawberry leaf with "Leaf Scorch" disease',
            'pesticide': 'Use fungicides like Myclobutanil or Captan.',
            'fertilizer': 'Vermicompost or farmyard manure.'
        },
        'Sugarcane___Healthy': {
            'definition': 'This is a healthy sugarcane leaf',
            'pesticide': 'No pesticide recommended as it is stated Healthy',
            'fertilizer': 'Organic fertilizers like compost, farmyard manure, and green manure.'
        },
        'Sugarcane___Mosaic': {
            'definition': 'This is a sugarcane leaf with "Mosaic" disease',
            'pesticide': 'Use resistant varieties and ensure good plant nutrition with organic fertilizers.',
            'fertilizer': 'Neem cake or compost.'
        },
        'Sugarcane___RedRot': {
            'definition': 'This is a sugarcane leaf with "Red Rot" disease',
            'pesticide': ' Focus on resistant varieties and balanced fertilization.',
            'fertilizer': ' Vermicompost or farmyard manure.'
        },
        'Sugarcane___Rust': {
            'definition': 'This is a sugarcane leaf with "Rust" disease',
            'pesticide': 'Use fungicides like Propiconazole.',
            'fertilizer': 'Green manure or compost  .'
        },
        'Sugarcane___Yellow': {
            'definition': 'This is a sugarcane leaf with "Yellow leaf" disease',
            'pesticide': 'Ensure good plant nutrition and use balanced organic fertilizers',
            'fertilizer': 'Compost or farmyard manure.'
        },
        'Tomato___Bacterial_spot': {
            'definition': 'This is a tomato leaf with "Bacterial Spot" disease',
            'pesticide': 'Use copper-based bactericides.',
            'fertilizer': 'Neem cake or vermicompost.'
        },
        'Tomato___Early_blight': {
            'definition': 'This is a tomato leaf with "Early Blight" disease',
            'pesticide': 'Use fungicides like Chlorothalonil or Mancozeb.',
            'fertilizer': 'Compost or farmyard manure.'
        },
        'Tomato___healthy': {
            'definition': 'This is a healthy tomato leaf',
            'pesticide': 'No pesticide recommended as it is stated Healthy',
            'fertilizer': 'Organic fertilizers like compost or vermicompost.'
        },
        'Tomato___Late_blight': {
            'definition': 'This is a tomato leaf with "Late Blight" disease',
            'pesticide': 'Use fungicides like Chlorothalonil or Mancozeb.',
            'fertilizer': ' Vermicompost or green manure.'
        },
        'Tomato___Leaf_Mold': {
            'definition': 'This is a tomato leaf with "Leaf Mold" disease',
            'pesticide': 'Use fungicides like Chlorothalonil or Copper-based products',
            'fertilizer': 'Compost or neem cake.'
        },
        'Tomato___Septoria_leaf_spot': {
            'definition': 'This is a tomato leaf with "Septoria Leaf Spot" disease',
            'pesticide': 'Use fungicides like Chlorothalonil or Mancozeb.',
            'fertilizer': 'Farmyard manure or compost.'
        },
        'Tomato___Spider_mites Two-spotted_spider_mite': {
            'definition': 'This is a tomato leaf with "Spider Mites" disease',
            'pesticide': 'Use miticides like Abamectin.',
            'fertilizer': 'Vermicompost or neem cake.'
        },
        'Tomato___Target_Spot': {
            'definition': 'This is a tomato leaf with "Target Spot" disease',
            'pesticide': 'Use fungicides like Chlorothalonil or Mancozeb.',
            'fertilizer': 'Compost or farmyard manure.'
        },
        'Tomato___Tomato_mosaic_virus': {
            'definition': 'This is a tomato leaf with "Tomato Mosaic Virus" disease',
            'pesticide': 'Ensure good plant nutrition with organic fertilizers and use resistant varieties.',
            'fertilizer': 'Vermicompost or compost.'
        },
        'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
            'definition': 'This is a tomato leaf with "Yellow Leaf Curl Virus" disease',
            'pesticide': 'Control whiteflies with insecticides like Imidacloprid.',
            'fertilizer': 'Neem cake or compost.'
        },
        'Wheat___Brown_Rust': {
            'definition': 'This is a rice leaf with "Leaf Blast" disease',
            'pesticide': 'Use fungicides like Propiconazole or Mancozeb.',
            'fertilizer': 'Farmyard manure or green manure.'
        },
        'Wheat___Healthy': {
            'definition': 'This is a healthy wheat leaf',
            'pesticide': 'No pesticide recommended as it is stated Healthy',
            'fertilizer': 'Organic fertilizers like compost or vermicompost.'
        },
        'Wheat___Yellow_Rust': {
            'definition': 'This is a wheat leaf with "Yellow Rust" disease',
            'pesticide': 'Use fungicides like Propiconazole or Tebuconazole',
            'fertilizer': 'Green manure or compost.'
        },
    }
    return recommendations.get(disease_name, {'definition':'No definition', 'pesticide': 'No recommendation', 'fertilizer': 'No recommendation'})
