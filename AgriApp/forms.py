# plant progress tracking
from django import forms
from .models import PlantGrowthRecord

# growth progress
class PlantImageUploadForm(forms.ModelForm):
    class Meta:
        model = PlantGrowthRecord
        fields = ['image']  # Only include the image field for upload
        widgets = {
            'image': forms.FileInput(attrs={'accept': 'image/*'}),
        }

class ReviewForm(forms.ModelForm):
    class Meta:
        model = PlantGrowthRecord
        fields = ['watering_instructions', 'fertilizer_instructions', 'pesticide_instructions']  # Include fields relevant for review

# price prediction
INDIAN_STATES = [
    ('AP', 'Andhra Pradesh'),
    ('AR', 'Arunachal Pradesh'),
    ('AS', 'Assam'),
    ('BR', 'Bihar'),
    ('CT', 'Chhattisgarh'),
    ('GA', 'Goa'),
    ('GJ', 'Gujarat'),
    ('HR', 'Haryana'),
    ('HP', 'Himachal Pradesh'),
    ('JH', 'Jharkhand'),
    ('KA', 'Karnataka'),
    ('KL', 'Kerala'),
    ('MP', 'Madhya Pradesh'),
    ('MH', 'Maharashtra'),
    ('MN', 'Manipur'),
    ('ML', 'Meghalaya'),
    ('MZ', 'Mizoram'),
    ('NL', 'Nagaland'),
    ('OR', 'Odisha'),
    ('PB', 'Punjab'),
    ('RJ', 'Rajasthan'),
    ('SK', 'Sikkim'),
    ('TN', 'Tamil Nadu'),
    ('TG', 'Telangana'),
    ('TR', 'Tripura'),
    ('UP', 'Uttar Pradesh'),
    ('UT', 'Uttarakhand'),
    ('WB', 'West Bengal'),
    ('AN', 'Andaman and Nicobar Islands'),
    ('CH', 'Chandigarh'),
    ('DN', 'Dadra and Nagar Haveli'),
    ('DD', 'Daman and Diu'),
    ('LD', 'Lakshadweep'),
    ('DL', 'Delhi'),
    ('PY', 'Puducherry')
]

class CropPriceForm(forms.Form):
    state = forms.ChoiceField(choices=INDIAN_STATES, label='State')
    crop = forms.CharField(label='Crop', max_length=100)
    cost_cultivation = forms.FloatField(label='Cost of Cultivation')
    cost_cultivation2 = forms.FloatField(label='Cost of Cultivation 2')
    production = forms.FloatField(label='Production')
    yield_ = forms.FloatField(label='Yield')
    temperature = forms.FloatField(label='Temperature')
    rain_fall_annual = forms.FloatField(label='Rain Fall Annual')