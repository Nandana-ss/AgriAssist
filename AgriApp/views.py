from django.shortcuts import render, HttpResponse, redirect
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import AuthenticationForm
from django.contrib import messages
import logging
from django.contrib.auth.decorators import login_required
import pickle
import os
from django.contrib.auth import get_user_model
from .core import model_prediction, get_recommendations
from django.core.files.storage import FileSystemStorage
import joblib
from django.http import JsonResponse
from PIL import Image
import numpy as np
from .models import PlantGrowthRecord
from tensorflow.keras.models import load_model
import pandas as pd
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from .forms import CropPriceForm

# Setup logger
logger = logging.getLogger(__name__)

# Views

def show_login(request):
    return render(request, 'login.html')

@login_required
def progress(request):
    return render(request, 'progress.html')

def index(request):
    return render(request, 'index.html', {'user': request.user})

def disease_recognition(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(image.name, image)
        file_url = fs.url(filename)
        result_index = model_prediction(fs.path(filename))

        class_name = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Bean___angular_leaf_spot', 'Bean___rust', 'Beans___healthy', 'Blueberry___healthy',
            'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn___Common_Rust',
            'Corn___Gray_Leaf_Spot', 'Corn___Healthy', 'Corn___Leaf_Blight', 'Grape___Black_rot',
            'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
            'Invalid', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
            'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_Blight', 'Potato___Healthy',
            'Potato___Late_Blight', 'Raspberry___healthy', 'Rice___Brown_Spot', 'Rice___Healthy', 'Rice___Hispa',
            'Rice___Leaf_Blast', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch',
            'Strawberry___healthy', 'Sugarcane___Healthy', 'Sugarcane___Mosaic', 'Sugarcane___RedRot',
            'Sugarcane___Rust', 'Sugarcane___Yellow', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
            'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
            'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy',
            'Wheat___Brown_Rust', 'Wheat___Healthy', 'Wheat___Yellow_Rust'
        ]

        disease_name = class_name[result_index]
        recommendations = get_recommendations(disease_name)

        context = {
            'file_url': file_url,
            'definition': recommendations.get('definition', 'No definition available'),
            'pesticide': recommendations.get('pesticide', 'No pesticide information available'),
            'fertilizer': recommendations.get('fertilizer', 'No fertilizer information available')
        }
        return render(request, 'disease_recognition.html', context)
    
    return render(request, 'disease_recognition.html')

@login_required(login_url='user_login')
def crop_recommendation(request):
    return render(request, 'crop_recommendation.html')

def analyze(request):
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models', 'cropRecommendation_model.pkl'))
    with open(model_path, 'rb') as model_file:
        RF = pickle.load(model_file)

    # Take values from user
    N = request.POST.get('nitrogen', '0')
    P = request.POST.get('phosphorous', '0')
    K = request.POST.get('potassium', '0')
    temp = request.POST.get('temperature', '0')
    humidity = request.POST.get('humidity', '0')
    ph = request.POST.get('ph', '0')
    rainfall = request.POST.get('rainfall', '0')

    # Ensure all inputs are converted to float
    try:
        userInput = [float(N), float(P), float(K), float(temp), float(humidity), float(ph), float(rainfall)]
    except ValueError:
        return HttpResponse("Invalid input values", status=400)

    # Use trained model to predict the data based on user input
    probabilities = RF.predict_proba([userInput])[0]

    # Get the top 3 crop predictions with the highest probabilities
    top_3_indices = probabilities.argsort()[-3:][::-1]
    top_3_crops = [RF.classes_[i] for i in top_3_indices]

    # Map crop names to image filenames
    crop_images = {
        'rice': 'rice.jpg',
        'maize': 'maize.jpg',
        'chickpea': 'chickpea.jpg',
        'kidneybeans': 'kidneybeans.jpg',
        'pigeonpeas': 'pigeonpeas.jpg',
        'mothbeans': 'mothbeans.jpg',
        'mungbean': 'mungbean.jpg',
        'blackgram': 'blackgram.jpg',
        'lentil': 'lentil.jpg',
        'pomegranate': 'pomegranate.jpg',
        'banana': 'banana.jpg',
        'mango': 'mango.jpg',
        'grapes': 'grapes.jpg',
        'watermelon': 'watermelon.jpg',
        'muskmelon': 'muskmelon.jpg',
        'apple': 'apple.jpg',
        'orange': 'orange.jpg',
        'papaya': 'papaya.jpg',
        'coconut': 'coconut.jpg',
        'cotton': 'cotton.jpg',
        'jute': 'jute.jpg',
        'coffee': 'coffee.jpg'
    }

    # Get the corresponding images for the top 3 crops
    top_3_crops_with_images = [(crop.upper(), crop_images.get(crop, 'default.jpg')) for crop in top_3_crops]

    # Display result to the user
    params = {
        'purpose': 'Top 3 Predicted Crops: ',
        'top_3_crops_with_images': top_3_crops_with_images
    }
    return render(request, 'crop_recommendation.html', params)

def register(request):
    if request.method == 'POST':
        uname = request.POST.get('username')
        email = request.POST.get('email')
        pass1 = request.POST.get('password1')
        pass2 = request.POST.get('password2')
        phone = request.POST.get('phone')

        if pass1 != pass2:
            return HttpResponse("Your password and confirm password do not match!")
        else:
            # Assuming you're using a custom user model with a phone field
            User = get_user_model()
            my_user = User.objects.create_user(username=uname, email=email, password=pass1)
            my_user.phone = phone  # Set the phone number
            my_user.save()
            return redirect('user_login')
        
    return render(request, 'register.html')

def user_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        pass1 = request.POST.get('password1')
        user = authenticate(request, username=username, password=pass1)
        if user is not None:
            login(request, user)
            return redirect('index')
        else:
            return HttpResponse("Username or Password is incorrect!!!")

    return render(request, 'login.html')

def logout_user(request):
    logout(request)
    messages.success(request, "You were logged out")
    return redirect('index')


from django.shortcuts import render
from django.http import JsonResponse
from .models import PlantGrowthRecord
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from PIL import Image
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import logging

# Setup logger
logger = logging.getLogger(__name__)

# Load models and other resources

image_model = load_model('AgriApp/models/plant_growth_model.keras')
csv_model = joblib.load('AgriApp/models/csv_growth_model.pkl')

# Define class names and care instructions
class_names = {
    0: 'Rice_seed_level',
    1: 'Rice_Germination_stage',
    2: 'Rice_Seedling_Stage',
    3: 'Rice_Grain_Filling_Stage',
    4: 'Rice_Dough_Stage',
    5: 'Rice_Harvesting_stage',
}

care_instructions = {
    'Rice_seed_level': {
        'Watering': 'Keep the soil moist but not waterlogged.',
        'Fertilizer': 'Use a balanced starter fertilizer.',
        'Pesticide': 'Monitor for pests and apply appropriate insecticide if necessary.',
    },
    'Rice_Germination_stage': {
        'Watering': 'Maintain constant moisture in the soil.',
        'Fertilizer': 'Apply a nitrogen-rich fertilizer to support early growth.',
        'Pesticide': 'Watch for fungal infections and treat as needed.',
    },
    'Rice_Seedling_Stage': {
        'Watering': 'Ensure the soil is consistently moist.',
        'Fertilizer': 'Use a balanced fertilizer with more nitrogen.',
        'Pesticide': 'Check for pests like aphids or leafhoppers.',
    },
    'Rice_Grain_Filling_Stage': {
        'Watering': 'Reduce watering as the grains start filling.',
        'Fertilizer': 'Apply a low-nitrogen fertilizer to avoid excessive growth.',
        'Pesticide': 'Monitor for diseases like leaf blast or bacterial blight.',
    },
    'Rice_Dough_Stage': {
        'Watering': 'Withhold water to allow grains to harden.',
        'Fertilizer': 'No additional fertilization needed.',
        'Pesticide': 'Ensure no fungal or pest issues are present.',
    },
    'Rice_Harvesting_stage': {
        'Watering': 'No watering needed.',
        'Fertilizer': 'No additional fertilization needed.',
        'Pesticide': 'Harvest timely to avoid pest infestations.',
    },
}

growth_stages = {
    'Rice_seed_level': 'Rice_Germination_stage',
    'Rice_Germination_stage': 'Rice_Seedling_Stage',
    'Rice_Seedling_Stage': 'Rice_Grain_Filling_Stage',
    'Rice_Grain_Filling_Stage': 'Rice_Dough_Stage',
    'Rice_Dough_Stage': 'Rice_Harvesting_stage',
    'Rice_Harvesting_stage':'No stages available'
}

# Define mappings for categorical features
soil_type_mapping = {'Sandy': 0, 'Clay': 1, 'Loamy': 2}
water_frequency_mapping = {'Daily': 0, 'Weekly': 1, 'Bi-Weekly': 2}
fertilizer_type_mapping = {'Organic': 0, 'Inorganic': 1, 'Balanced': 2}

@csrf_exempt
@require_http_methods(["POST"])
def predict_plant_growth(request):
    try:
        # Extract and map CSV data from request
        csv_data = {
            'Soil_Type': request.POST.get('Soil_Type'),
            'Sunlight_Hours': request.POST.get('Sunlight_Hours'),
            'Water_Frequency': request.POST.get('Water_Frequency'),
            'Fertilizer_Type': request.POST.get('Fertilizer_Type'),
            'Temperature': request.POST.get('Temperature'),
            'Humidity': request.POST.get('Humidity'),
        }
        # Validate the input fields
        if not all([csv_data['Soil_Type'], csv_data['Sunlight_Hours'], csv_data['Water_Frequency'], csv_data['Fertilizer_Type'], csv_data['Temperature'], csv_data['Humidity']]):
            return JsonResponse({'error': 'Missing required fields'}, status=400)
        soil_type = soil_type_mapping.get(csv_data['Soil_Type'], soil_type_mapping['Sandy'])
        water_frequency = water_frequency_mapping.get(csv_data['Water_Frequency'], water_frequency_mapping['Daily'])
        fertilizer_type = fertilizer_type_mapping.get(csv_data['Fertilizer_Type'], fertilizer_type_mapping['Organic'])
        
        csv_data_encoded = {
            'Soil_Type': soil_type,
            'Sunlight_Hours': float(csv_data.get('Sunlight_Hours', 0)),
            'Water_Frequency': water_frequency,
            'Fertilizer_Type': fertilizer_type,
            'Temperature': float(csv_data.get('Temperature', 0)),
            'Humidity': float(csv_data.get('Humidity', 0)),
        }

        csv_input = pd.DataFrame([csv_data_encoded])

        if csv_input.isnull().values.any():
            logger.error("CSV input contains NaN values.")
            return JsonResponse({'error': 'CSV input contains null values'}, status=400)

        # Predict the next growth stage using CSV model
        csv_prediction = csv_model.predict(csv_input)
        next_growth_stage_idx = int(csv_prediction[0])
        next_growth_stage = class_names.get(next_growth_stage_idx, 'Unknown')

        # Handle image input if applicable
        if 'image' in request.FILES:
            try:
                img_file = request.FILES['image']
                img = Image.open(img_file)
                 # Convert the image to RGB if it has an alpha channel
                if img.mode == 'RGBA':
                  img = img.convert('RGB')
                elif img.mode != 'RGB':
                  img = img.convert('RGB')
                img = img.resize((150, 150))
                img_array = np.expand_dims(np.array(img), axis=0) / 255.0

                img_prediction = image_model.predict(img_array)
                current_growth_stage_idx = np.argmax(img_prediction, axis=1)[0]
                current_growth_stage = class_names.get(current_growth_stage_idx, 'Unknown')
                instructions = care_instructions.get(current_growth_stage, {})
                
                
            except Exception as e:
                logger.error(f"Image prediction error: {str(e)}")
                return JsonResponse({'error': f'Image prediction error: {str(e)}'}, status=500)
        else:
            current_growth_stage = 'Unknown'
            instructions = {}

        # Save record in database
        try:
            record = PlantGrowthRecord(
                user=request.user,
                image=request.FILES.get('image') if 'image' in request.FILES else None,
                growth_stage=current_growth_stage,
                watering_instructions=instructions.get('Watering', 'No instructions available.'),
                fertilizer_instructions=instructions.get('Fertilizer', 'No instructions available.'),
                pesticide_instructions=instructions.get('Pesticide', 'No instructions available.'),
                next_growth_stage=next_growth_stage,
                soil_type=csv_data['Soil_Type'],
                water_frequency=csv_data['Water_Frequency'],
                fertilizer_type=csv_data['Fertilizer_Type'],
                additional_details=request.POST.get('additional_details', ''),
                
            )
            record.save()
        except Exception as e:
            logger.error(f"Error saving record: {str(e)}")
            return JsonResponse({'error': f'Error saving record: {str(e)}'}, status=500)
        # Fetch the care instructions for the next growth stage if available
        next_stage_instructions = care_instructions.get(next_growth_stage, {
         'Watering': 'No instructions available for next stage.',
         'Fertilizer': 'No instructions available for next stage.',
         'Pesticide': 'No instructions available for next stage.',
    })
        
        
        # Return success response
        response_data = {
            'current_growth_stage': current_growth_stage,
            'watering_instructions': instructions.get('Watering', 'No instructions available.'),
            'fertilizer_instructions': instructions.get('Fertilizer', 'No instructions available.'),
            'pesticide_instructions': instructions.get('Pesticide', 'No instructions available.'),
            'next_growth_stage': growth_stages.get(current_growth_stage, 'No further stages'),
            'next_watering_instructions': next_stage_instructions.get('Watering', 'No instructions available for next stage.'),
            'next_fertilizer_instructions': next_stage_instructions.get('Fertilizer', 'No instructions available for next stage.'),
            'next_pesticide_instructions': next_stage_instructions.get('Pesticide', 'No instructions available for next stage.'),
        }
        return JsonResponse(response_data)
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return JsonResponse({'error': f'Unexpected error: {str(e)}'}, status=500)

# email
from django.shortcuts import render
from django.http import HttpResponse
from .tasks import update_growth_stage_and_send_reminders
from django_celery_beat.models import PeriodicTask, CrontabSchedule
from datetime import datetime

# Create your views here.

def trigger_task_now(request):
    """
    Trigger the Celery task immediately.
    """
    update_growth_stage_and_send_reminders.delay()
    return HttpResponse("Task triggered immediately!")

def testView(request):
    update_growth_stage_and_send_reminders.delay()
    return HttpResponse("Done")

def send_mail_to_all_users(request):
    update_growth_stage_and_send_reminders.delay()
    return HttpResponse("Email has been sent successfully")

def sendmailattime(request):
    # Create a schedule for the task to run at a specific time
    schedule, created = CrontabSchedule.objects.get_or_create(hour=11, minute=50)

    # Create or update a periodic task
    task, created = PeriodicTask.objects.get_or_create(
        crontab=schedule,
        name="update_growth_stage_task",
        defaults={'task': 'your_app.tasks.update_growth_stage_and_send_reminders'}
    )

    if not created:
        task.task = 'your_app.tasks.update_growth_stage_and_send_reminders'
        task.save()

    return HttpResponse("Success")


# price
from django.shortcuts import render
from .forms import CropPriceForm
import pickle
import pandas as pd
from django.conf import settings

# Load the model and preprocessing objects
model_path = os.path.join(settings.BASE_DIR, 'AgriApp', 'models', 'price_prediction_model.pkl')
with open(model_path, 'rb') as file:
    model_data = pickle.load(file)
    model = model_data['model']
    imputer = model_data['imputer']
    label_encoder_state = model_data['label_encoder_state']
    label_encoder_crop = model_data['label_encoder_crop']

def predict_price(request):
    if request.method == 'POST':
        form = CropPriceForm(request.POST)
        if form.is_valid():
            state = form.cleaned_data['state']
            crop = form.cleaned_data['crop']
            cost_cultivation = form.cleaned_data['cost_cultivation']
            cost_cultivation2 = form.cleaned_data['cost_cultivation2']
            production = form.cleaned_data['production']
            yield_ = form.cleaned_data['yield_']
            temperature = form.cleaned_data['temperature']
            rain_fall_annual = form.cleaned_data['rain_fall_annual']

            # Encode the input data
            if state in label_encoder_state.classes_:
                state_encoded = label_encoder_state.transform([state])[0]
            else:
                state_encoded = -1  # Handle unseen labels appropriately

            if crop in label_encoder_crop.classes_:
                crop_encoded = label_encoder_crop.transform([crop])[0]
            else:
                crop_encoded = -1  # Handle unseen labels appropriately

            new_data = pd.DataFrame({
                'State': [state_encoded],
                'Crop': [crop_encoded],
                'CostCultivation': [cost_cultivation],
                'CostCultivation2': [cost_cultivation2],
                'Production': [production],
                'Yield': [yield_],
                'Temperature': [temperature],
                'RainFall Annual': [rain_fall_annual]
            })

            new_data_imputed = imputer.transform(new_data)
            predicted_price = model.predict(new_data_imputed)[0]

            return render(request, 'result.html', {'price': predicted_price})

    else:
        form = CropPriceForm()

    return render(request, 'price_prediction.html', {'form': form})

def fetch_user_history(request):
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        records = PlantGrowthRecord.objects.filter(user=request.user).order_by('-date_uploaded')
        records_list = list(records.values(
            'image',
            'growth_stage',
            'watering_instructions',
            'fertilizer_instructions',
            'pesticide_instructions',
            'next_growth_stage',
            'date_uploaded'
        ))
        return JsonResponse({'records': records_list})
    return JsonResponse({'error': 'Invalid request'}, status=400)  

# @login_required
# def profile(request):
#     return render(request, 'profile.html') 

from django.shortcuts import render

def profile(request):
    return render(request, 'profile.html')