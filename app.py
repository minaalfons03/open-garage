import os
import io
import base64
import json
import tempfile
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import cv2
import firebase_admin
from firebase_admin import credentials, db as firebase_db
from datetime import datetime

app = Flask(__name__)
CORS(app)

# ── Firebase ──────────────────────────────────────────────────────────
firebase_ref = None

def init_firebase():
    global firebase_ref
    try:
        creds_json = os.environ.get("FIREBASE_CREDENTIALS", "")
        if not creds_json:
            print("⚠ FIREBASE_CREDENTIALS not set")
            return
        creds_json = creds_json.replace('\\n', '\n')
        creds_dict = json.loads(creds_json)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(creds_dict, f)
            creds_path = f.name
        cred = credentials.Certificate(creds_path)
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://garage-6749f-default-rtdb.firebaseio.com/'
        })
        firebase_ref = firebase_db.reference('/')
        print("✓ Firebase connected")
    except Exception as e:
        print(f"✗ Firebase init error: {e}")

init_firebase()

# ── OpenCV ────────────────────────────────────────────────────────────
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
recognizer   = cv2.face.LBPHFaceRecognizer_create()

FACES_DIR = "/tmp/known_faces"
os.makedirs(FACES_DIR, exist_ok=True)

label_map = {}  # int  → name
name_map  = {}  # name → int

def decode_image(b64_str):
    img_bytes = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    if max(img.size) > 640:
        img.thumbnail((640, 640), Image.LANCZOS)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

def extract_face(gray):
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return cv2.resize(gray[y:y+h, x:x+w], (200, 200))

def retrain():
    global recognizer, label_map, name_map
    faces_data, labels_data = [], []
    label_map, name_map = {}, {}
    counter = 0
    for fname in os.listdir(FACES_DIR):
        if not fname.endswith(".jpg"):
            continue
        name = fname[:-4]
        gray = cv2.imread(os.path.join(FACES_DIR, fname), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        face = extract_face(gray)
        if face is None:
            continue
        if name not in name_map:
            name_map[name] = counter
            label_map[counter] = name
            counter += 1
        faces_data.append(face)
        labels_data.append(name_map[name])
    if faces_data:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces_data, np.array(labels_data))
        print(f"✓ Trained on {len(faces_data)} face(s): {list(name_map.keys())}")
    else:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        print("⚠ No faces to train on")

# ── Firebase face persistence ─────────────────────────────────────────
def save_face_to_firebase(name, img_path):
    if not firebase_ref:
        return
    try:
        with open(img_path, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')
        firebase_ref.child('faces_data').child(name).set({'image': b64})
        print(f"✓ Saved {name} to Firebase")
    except Exception as e:
        print(f"⚠ Firebase save failed: {e}")

def load_faces_from_firebase():
    if not firebase_ref:
        return
    try:
        data = firebase_ref.child('faces_data').get()
        if not data:
            print("⚠ No faces in Firebase yet")
            return
        for name, val in data.items():
            if not val or 'image' not in val:
                continue
            img_bytes = base64.b64decode(val['image'])
            img_path = os.path.join(FACES_DIR, f"{name}.jpg")
            with open(img_path, 'wb') as f:
                f.write(img_bytes)
            print(f"↺ Loaded: {name}")
    except Exception as e:
        print(f"⚠ Firebase load failed: {e}")

def delete_face_from_firebase(name):
    if not firebase_ref:
        return
    try:
        firebase_ref.child('faces_data').child(name).delete()
        print(f"✓ Deleted {name} from Firebase")
    except Exception as e:
        print(f"⚠ Firebase delete failed: {e}")

def log_access(name, status, confidence):
    if not firebase_ref:
        return
    try:
        entry = {
            'name': name,
            'status': status,
            'confidence': confidence,
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        firebase_ref.child('garage/access_log').push(entry)
    except Exception as e:
        print(f"⚠ Log failed: {e}")

# ── Startup ───────────────────────────────────────────────────────────
print("↺ Loading faces from Firebase...")
load_faces_from_firebase()
retrain()

# ── Routes ────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "registered_faces": list(name_map.keys()),
        "firebase": "connected" if firebase_ref else "disconnected"
    })

@app.route("/register", methods=["POST"])
def register_face():
    data = request.get_json()
    name = data.get("name", "").strip()
    image_b64 = data.get("image", "")
    if not name or not image_b64:
        return jsonify({"error": "name and image required"}), 400
    try:
        gray = decode_image(image_b64)
        face = extract_face(gray)
        if face is None:
            return jsonify({"error": "No face detected. Use a clear well-lit photo."}), 422
        img_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        if max(img.size) > 640:
            img.thumbnail((640, 640), Image.LANCZOS)
        img_path = os.path.join(FACES_DIR, f"{name}.jpg")
        img.save(img_path, quality=95)
        save_face_to_firebase(name, img_path)
        retrain()
        print(f"✓ Registered: {name}")
        return jsonify({"success": True, "name": name, "total": len(name_map)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/garage/recognise", methods=["POST"])
def garage_recognise():
    data = request.get_json()
    image_b64 = data.get("image", "")
    if not image_b64:
        return jsonify({"error": "image required"}), 400
    if not name_map:
        return jsonify({"name": None, "status": "no_faces", "door": "closed"})
    try:
        gray = decode_image(image_b64)
        face = extract_face(gray)

        if face is None:
            if firebase_ref:
                firebase_ref.child('garage').update({
                    'current_visitor': 'Unknown',
                    'door_status': 'closed'
                })
            return jsonify({"name": None, "status": "no_face", "door": "closed", "faces_found": 0})

        label, distance = recognizer.predict(face)
        name = label_map.get(label, "Unknown")
        confidence = round(max(0, (100 - distance) / 100), 3)
        print(f"→ {name} | distance={distance:.1f} | confidence={confidence}")

        if distance > 100:
            name = "Unknown"
            confidence = 0

        if name != "Unknown":
            if firebase_ref:
                firebase_ref.child('garage').update({
                    'current_visitor': name,
                    'door_status': 'open',
                    'last_seen': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
            log_access(name, 'granted', confidence)
            print(f"✓ Access GRANTED: {name}")
            return jsonify({"name": name, "confidence": confidence, "door": "open", "status": "granted", "faces_found": 1})
        else:
            if firebase_ref:
                firebase_ref.child('garage').update({
                    'current_visitor': 'Unknown',
                    'door_status': 'closed'
                })
            log_access('Unknown', 'denied', confidence)
            print("✗ Access DENIED: Unknown")
            return jsonify({"name": None, "confidence": 0, "door": "closed", "status": "denied", "faces_found": 1})

    except Exception as e:
        print(f"Recognise error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/garage/close", methods=["POST"])
def garage_close():
    if firebase_ref:
        firebase_ref.child('garage').update({'door_status': 'closed'})
    return jsonify({"door": "closed"})

@app.route("/faces", methods=["GET"])
def list_faces():
    return jsonify({"faces": list(name_map.keys())})

@app.route("/faces/<name>", methods=["DELETE"])
def delete_face(name):
    if name in name_map:
        try:
            os.remove(os.path.join(FACES_DIR, f"{name}.jpg"))
        except:
            pass
        delete_face_from_firebase(name)
        retrain()
        return jsonify({"success": True, "deleted": name})
    return jsonify({"error": "Not found"}), 404

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
