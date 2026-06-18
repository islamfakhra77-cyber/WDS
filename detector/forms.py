from django import forms
from django.forms.widgets import ClearableFileInput


class ImageUploadForm(forms.Form):
    """Form for uploading an image for weapon detection."""
    image = forms.ImageField(
        label='Select an image to scan',
        widget=ClearableFileInput(attrs={
            'class': 'form-control',
            'accept': 'image/*',
            'id': 'image-upload',
        }),
        help_text='Supported formats: JPG, JPEG, PNG, BMP'
    )
