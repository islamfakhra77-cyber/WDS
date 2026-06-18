"""
============================================================
  VIEWS - Weapon Detection System (YOLOv8 + TF Model)
  Image + Video + Webcam detection
============================================================
"""

import os
import json
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.conf import settings
from django.core.files.storage import default_storage
from django.views.decorators.csrf import csrf_exempt
from .forms import ImageUploadForm
from .models import DetectionResult
from .detection_engine import predict_image, predict_frame_base64, predict_video, get_model_info


def home(request):
    """Home page with file upload + webcam."""
    model_info = get_model_info()
    recent = DetectionResult.objects.all()[:5] 
    
    context = {
        'form': ImageUploadForm(),
        'model_loaded': model_info.get('loaded', False),
        'model_info': model_info,
        'recent_detections': recent,
        'total_detections': DetectionResult.objects.count(),
        'weapon_count': DetectionResult.objects.filter(is_weapon=True).count(),
    }
    return render(request, 'detector/home.html', context)


def detect_image(request):
    """Handle image file upload and detect weapons."""
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES['image']
            file_path = default_storage.save(
                os.path.join('uploads', uploaded_file.name), uploaded_file
            )
            full_path = os.path.join(settings.MEDIA_ROOT, file_path)
            
            result = predict_image(full_path)
            
            detection = DetectionResult.objects.create(
                image=file_path,
                is_weapon=result['is_weapon'],
                confidence=result['confidence'],
                predicted_label=result['predicted_label'],
            )
            
            context = {
                'form': ImageUploadForm(),
                'result': result,
                'detection': detection,
                'image_url': settings.MEDIA_URL + file_path,
                'model_loaded': True,
            }
            return render(request, 'detector/result.html', context)
    
    return redirect('home')


@csrf_exempt
def detect_api(request):
    """API endpoint for image file upload detection (AJAX)."""
    if request.method == 'POST':
        if 'image' not in request.FILES:
            return JsonResponse({'error': 'No image provided'}, status=400)
        
        uploaded_file = request.FILES['image']
        file_path = default_storage.save(
            os.path.join('uploads', uploaded_file.name), uploaded_file
        )
        full_path = os.path.join(settings.MEDIA_ROOT, file_path)
        
        result = predict_image(full_path)
        
        DetectionResult.objects.create(
            image=file_path,
            is_weapon=result['is_weapon'],
            confidence=result['confidence'],
            predicted_label=result['predicted_label'],
        )
        
        return JsonResponse({
            'is_weapon': result['is_weapon'],
            'confidence': result['confidence'],
            'predicted_label': result['predicted_label'],
            'weapon_probability': result['weapon_probability'],
            'weapon_detections': result.get('weapon_detections', []),
            'all_detections': result.get('all_detections', []),
            'annotated_image': result.get('annotated_image'),
            'image_url': settings.MEDIA_URL + file_path,
        })
    
    return JsonResponse({'error': 'Only POST allowed'}, status=405)


@csrf_exempt
def detect_frame(request):
    """API endpoint for real-time webcam frame detection (YOLO + TF)."""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            image_b64 = data.get('image', '')
            
            if not image_b64:
                return JsonResponse({'error': 'No image data'}, status=400)
            
            result = predict_frame_base64(image_b64)
            return JsonResponse(result)
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Only POST allowed'}, status=405)


@csrf_exempt
def detect_video_api(request):
    """API endpoint for VIDEO upload detection."""
    if request.method == 'POST':
        if 'video' not in request.FILES:
            return JsonResponse({'error': 'No video provided'}, status=400)
        
        try:
            uploaded_video = request.FILES['video']
            # Save video
            video_path = default_storage.save(
                os.path.join('uploads', uploaded_video.name), uploaded_video
            )
            full_path = os.path.join(settings.MEDIA_ROOT, video_path)
            
            print(f"Processing video: {full_path}")
            
            # Process video
            result = predict_video(full_path, frame_skip=5)
            
            # Save result
            DetectionResult.objects.create(
                image=video_path,
                is_weapon=result.get('is_weapon_in_video', False),
                confidence=result.get('weapon_frames', [{}])[0].get('confidence', 0) if result.get('weapon_frames') else 0,
                predicted_label='Weapon Detected' if result.get('is_weapon_in_video') else 'No Weapon',
            )
            
            result['video_url'] = settings.MEDIA_URL + video_path
            return JsonResponse(result)
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Only POST allowed'}, status=405)


def history(request):
    """View detection history."""
    detections = DetectionResult.objects.all()
    context = {
        'detections': detections,
        'total': detections.count(),
        'weapon_count': detections.filter(is_weapon=True).count(),
        'safe_count': detections.filter(is_weapon=False).count(),
    }
    return render(request, 'detector/history.html', context)


def about(request):
    """About page with model info."""
    model_info = get_model_info()
    context = {
        'model_info': model_info,
        'model_loaded': model_info.get('loaded', False),
    }
    return render(request, 'detector/about.html', context)

 