from django import forms
from crop_api import ProfessionalCropRecommender

class RecommendationForm(forms.Form):
    soil_ph = forms.FloatField(
        label="Soil pH",
        min_value=3.5,
        max_value=9.5,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'})
    )
    soil_temp = forms.FloatField(
        label="Soil Temperature (°C)",
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    soil_type = forms.ChoiceField(
        label="Soil Type",
        choices=[(t, t) for t in ProfessionalCropRecommender.VALID_SOIL_TYPES],
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    rainfall = forms.IntegerField(
        label="Annual Rainfall (mm)",
        min_value=0,
        max_value=5000,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    humidity = forms.FloatField(
        label="Average Humidity (%)",
        min_value=0,
        max_value=100,
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    location = forms.CharField(
        label="Location (optional)",
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )