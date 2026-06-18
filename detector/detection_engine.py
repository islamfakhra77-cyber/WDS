"""
============================================================
  WEAPON DETECTION ENGINE - TF Model (FAST) + YOLO (boxes)
  Webcam = SIRF TF Model (jaise script mein tha - FAST!)
  Image = TF + YOLO dono
  Video = TF + YOLO dono
============================================================
"""

import os
import cv2
import numpy as np
import base64
from PIL import Image
from ultralytics import YOLO

# YOLO models
_yolo_model = None
_custom_yolo = None
WEAPON_CLASSES = {43: "knife", 76: "scissors", 34: "baseball bat"}

# TensorFlow model
_tf_model = None

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YOLO_MODEL_PATH = os.path.join(BASE_DIR, 'yolov8n.pt')
CUSTOM_YOLO_PATH = os.path.join(BASE_DIR, 'models', 'weapon_best.pt')
TF_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'weapon_detector.h5')


def load_yolo_model():
    global _yolo_model
    if _yolo_model is None:
        if os.path.exists(YOLO_MODEL_PATH):
            _yolo_model = YOLO(YOLO_MODEL_PATH)
        else:
            _yolo_model = YOLO('yolov8n.pt')
        print("YOLOv8n model loaded!")
    return _yolo_model


def load_custom_yolo():
    global _custom_yolo
    if os.path.exists(CUSTOM_YOLO_PATH):
        if _custom_yolo is None:
            _custom_yolo = YOLO(CUSTOM_YOLO_PATH)
            print("Custom YOLO weapon model loaded!")
        return _custom_yolo
    return None


def load_tf_model():
    """Load tera trained TensorFlow model."""
    global _tf_model
    if _tf_model is None:
        if os.path.exists(TF_MODEL_PATH):
            try:
                import tensorflow as tf
                print(f"Loading TF model from {TF_MODEL_PATH}...")
                _tf_model = tf.keras.models.load_model(TF_MODEL_PATH)
                print("TF model loaded! (tera trained model)")
            except Exception as e:
                print(f"TF model load failed: {e}")
        else:
            print(f"TF model not found at {TF_MODEL_PATH}")
    return _tf_model


def _tf_predict_frame(img):
    """TF model se single frame check karo - FAST."""
    tf_model = load_tf_model()
    if tf_model is None:
        return False, 0.0
    try:
        import tensorflow as tf
        resized = cv2.resize(img, (224, 224))
        img_array = resized.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = tf_model.predict(img_array, verbose=0)
        weapon_prob = float(prediction[0][0])
        is_weapon = weapon_prob > 0.6
        confidence = weapon_prob * 100 if is_weapon else (1 - weapon_prob) * 100
        return is_weapon, confidence
    except Exception as e:
        print(f"TF frame predict error: {e}")
        return False, 0.0


def predict_image(image_path):
    """Image upload - TF classification + YOLO boxes."""
    yolo = load_yolo_model()
    custom = load_custom_yolo()
    tf_model = load_tf_model()

    try:
        img = cv2.imread(image_path)
        if img is None:
            return {
                'is_weapon': False, 'confidence': 0.0,
                'predicted_label': 'Error', 'weapon_probability': 0.0,
                'error': 'Cannot read image',
                'weapon_detections': [], 'all_detections': [],
                'annotated_image': None
            }

        # TF Classification PEHLE
        tf_weapon = False
        tf_confidence = 0.0
        if tf_model is not None:
            try:
                import tensorflow as tf
                test_img = Image.open(image_path).resize((224, 224))
                img_array = np.array(test_img)
                if img_array.ndim == 2:
                    img_array = np.stack([img_array]*3, axis=-1)
                elif img_array.shape[2] == 4:
                    img_array = img_array[:,:,:3]
                img_array = np.expand_dims(img_array, axis=0) / 255.0

                prediction = tf_model.predict(img_array, verbose=0)
                weapon_prob = float(prediction[0][0])
                tf_weapon = weapon_prob > 0.6
                tf_confidence = weapon_prob * 100 if tf_weapon else (1 - weapon_prob) * 100
                print(f"TF Model: weapon_prob={weapon_prob:.4f}, is_weapon={tf_weapon}, conf={tf_confidence:.1f}%")
            except Exception as e:
                print(f"TF prediction error: {e}")

        # YOLO Detection (bounding boxes)
        if custom:
            results = custom(img, conf=0.25, verbose=False)
            weapon_dets, all_dets = _process_custom(results)
        else:
            results = yolo(img, conf=0.25, verbose=False)
            weapon_dets, all_dets = _process_default(results)

        # Combine
        is_weapon = len(weapon_dets) > 0 or tf_weapon
        yolo_conf = max([d['confidence'] for d in weapon_dets], default=0)
        confidence = max(yolo_conf, tf_confidence)

        if tf_weapon and len(weapon_dets) == 0:
            h, w = img.shape[:2]
            weapon_dets.append({
                'class': 'weapon',
                'confidence': round(tf_confidence, 2),
                'bbox': {'x1': int(w*0.1), 'y1': int(h*0.1), 'x2': int(w*0.9), 'y2': int(h*0.9)},
                'is_weapon': True,
                'source': 'tensorflow'
            })
            all_dets.append(weapon_dets[-1])

        annotated = _draw_detections(img.copy(), weapon_dets, all_dets)
        _, buf = cv2.imencode('.jpg', annotated)
        img_b64 = base64.b64encode(buf).decode('utf-8')

        return {
            'is_weapon': is_weapon,
            'confidence': round(confidence, 2),
            'predicted_label': 'Weapon' if is_weapon else 'No Weapon',
            'weapon_probability': round(confidence / 100, 4),
            'weapon_detections': weapon_dets,
            'all_detections': all_dets,
            'annotated_image': img_b64,
            'image_size': {'width': img.shape[1], 'height': img.shape[0]},
            'model_used': 'TF+YOLO' if tf_model else 'YOLO',
            'error': None
        }
    except Exception as e:
        return {
            'is_weapon': False, 'confidence': 0.0,
            'predicted_label': 'Error', 'weapon_probability': 0.0,
            'error': str(e),
            'weapon_detections': [], 'all_detections': [],
            'annotated_image': None
        }


def predict_frame_base64(image_b64):
    """Webcam frame - SIRF TF Model (FAST like script!)"""
    tf_model = load_tf_model()

    try:
        if ',' in image_b64:
            image_b64 = image_b64.split(',')[1]

        img_bytes = base64.b64decode(image_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {'weapon_detected': False, 'weapon_count': 0, 'detections': [], 'error': 'Bad image'}

        h, w = img.shape[:2]
        if w > 640:
            img = cv2.resize(img, (640, int(h * 640 / w)))

        h, w = img.shape[:2]

        # === SIRF TF MODEL - Jaise script mein tha! ===
        tf_weapon = False
        tf_confidence = 0.0
        if tf_model is not None:
            tf_weapon, tf_confidence = _tf_predict_frame(img)

        # === Red Box draw karo ===
        annotated_img = img.copy()
        if tf_weapon:
            cv2.rectangle(annotated_img, (int(w*0.05), int(h*0.05)), (int(w*0.95), int(h*0.95)), (0, 0, 255), 4)
            label = f"WEAPON: {tf_confidence:.0f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(annotated_img, (10, 10), (10+tw+10, 10+th+15), (0, 0, 255), -1)
            cv2.putText(annotated_img, label, (15, 10+th+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.putText(annotated_img, "DANGER!", (int(w/2)-80, int(h/2)),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        else:
            label = f"SAFE: {(100-tf_confidence):.0f}%"
            cv2.putText(annotated_img, label, (15, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        _, buf = cv2.imencode('.jpg', annotated_img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        annotated_b64 = base64.b64encode(buf).decode('utf-8')

        weapon_dets = []
        if tf_weapon:
            weapon_dets.append({
                'class': 'WEAPON',
                'confidence': round(tf_confidence, 2),
                'bbox': {'x1': int(w*0.05), 'y1': int(h*0.05), 'x2': int(w*0.95), 'y2': int(h*0.95)},
                'is_weapon': True,
                'source': 'tensorflow'
            })

        return {
            'weapon_detected': tf_weapon,
            'weapon_count': 1 if tf_weapon else 0,
            'weapon_detections': weapon_dets,
            'all_detections': weapon_dets,
            'total_objects': 1 if tf_weapon else 0,
            'tf_model_detected': tf_weapon,
            'tf_confidence': round(tf_confidence, 2),
            'confidence': round(tf_confidence, 2),
            'annotated_frame': annotated_b64,
            'error': None
        }
    except Exception as e:
        return {'weapon_detected': False, 'weapon_count': 0, 'detections': [], 'error': str(e)}


def predict_video(video_path, frame_skip=5):
    """Video file se weapon detect karo."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {'error': 'Cannot open video', 'weapon_frames': [], 'total_frames': 0}

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    yolo = load_yolo_model()
    custom = load_custom_yolo()
    tf_model = load_tf_model()

    weapon_frames = []
    frame_count = 0
    checked = 0

    print(f"Processing video: {width}x{height} @ {fps}fps, {total} frames")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % frame_skip != 0:
            continue

        checked += 1

        h, w = frame.shape[:2]
        if w > 640:
            frame = cv2.resize(frame, (640, int(h * 640 / w)))

        # TF check PEHLE
        tf_weapon = False
        tf_conf = 0.0
        if tf_model is not None:
            tf_weapon, tf_conf = _tf_predict_frame(frame)

        # YOLO check
        if custom:
            results = custom(frame, conf=0.25, verbose=False)
            weapon_dets, _ = _process_custom(results)
        else:
            results = yolo(frame, conf=0.25, verbose=False)
            weapon_dets, _ = _process_default(results)

        is_weapon = len(weapon_dets) > 0 or tf_weapon
        max_conf = max([d['confidence'] for d in weapon_dets], default=tf_conf)

        if is_weapon:
            timestamp = round(frame_count / fps, 2)
            weapon_frames.append({
                'frame': frame_count,
                'timestamp': timestamp,
                'confidence': round(max_conf, 2),
                'yolo_detected': len(weapon_dets) > 0,
                'tf_detected': tf_weapon,
                'detections': weapon_dets if weapon_dets else [{'class': 'weapon', 'confidence': round(tf_conf, 2), 'source': 'tensorflow'}]
            })

    cap.release()

    print(f"Video done: {checked} frames checked, {len(weapon_frames)} weapon detections")

    return {
        'total_frames': frame_count,
        'frames_checked': checked,
        'weapon_frame_count': len(weapon_frames),
        'weapon_frames': weapon_frames,
        'is_weapon_in_video': len(weapon_frames) > 0,
        'video_info': {'fps': fps, 'width': width, 'height': height, 'duration': round(frame_count/fps, 2)},
        'error': None
    }


def _process_default(results):
    weapon_dets = []
    all_dets = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            name = result.names[cls_id]
            det = {
                'class': name, 'confidence': round(conf * 100, 2),
                'bbox': {'x1': round(x1,1), 'y1': round(y1,1), 'x2': round(x2,1), 'y2': round(y2,1)},
                'is_weapon': cls_id in WEAPON_CLASSES
            }
            all_dets.append(det)
            if det['is_weapon']:
                weapon_dets.append(det)
    return weapon_dets, all_dets


def _process_custom(results):
    weapon_dets = []
    all_dets = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            name = result.names[cls_id]
            det = {
                'class': name, 'confidence': round(conf * 100, 2),
                'bbox': {'x1': round(x1,1), 'y1': round(y1,1), 'x2': round(x2,1), 'y2': round(y2,1)},
                'is_weapon': True
            }
            all_dets.append(det)
            weapon_dets.append(det)
    return weapon_dets, all_dets


def _draw_detections(img, weapon_dets, all_dets):
    for det in all_dets:
        if not det['is_weapon']:
            b = det['bbox']
            cv2.rectangle(img, (int(b['x1']),int(b['y1'])), (int(b['x2']),int(b['y2'])), (0,255,0), 2)
            label = f"{det['class']} {det['confidence']}%"
            cv2.putText(img, label, (int(b['x1']),int(b['y1'])-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    for det in weapon_dets:
        b = det['bbox']
        x1, y1, x2, y2 = int(b['x1']), int(b['y1']), int(b['x2']), int(b['y2'])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 4)
        src = det.get('source', 'yolo')
        label = f"WEAPON({src}): {det['class']} {det['confidence']}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1-th-10), (x1+tw+10, y1), (0, 0, 255), -1)
        cv2.putText(img, label, (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    if len(weapon_dets) > 0:
        h, w = img.shape[:2]
        cv2.putText(img, "DANGER!", (int(w/2)-80, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

    return img


def get_model_info():
    yolo = load_yolo_model()
    custom = load_custom_yolo()
    tf_model = load_tf_model()
    return {
        'loaded': yolo is not None,
        'default_model': 'yolov8n (COCO 80 classes)',
        'custom_model': 'loaded' if custom else 'not available',
        'tf_model': 'loaded (tera trained model)' if tf_model else 'not found',
        'tf_model_path': TF_MODEL_PATH,
        'weapon_classes': list(WEAPON_CLASSES.values()),
        'engine': 'YOLOv8 + TensorFlow'
    }