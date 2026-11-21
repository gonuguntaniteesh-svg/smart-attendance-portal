import os
import json
import joblib
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import ViTImageProcessor, ViTModel
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "face_dataset")

class FaceRecognitionEngine:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "google/vit-base-patch16-224-in21k"
        self.confidence_threshold = 0.65
        
        # Load Haar Cascade Face Detector with safe fallback
        local_cascade = os.path.join(MODELS_DIR, "haarcascade_frontalface_default.xml")
        if os.path.exists(local_cascade):
            cascade_path = local_cascade
        elif hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        else:
            cascade_path = "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # ViT Processor & Model (Lazy initialized)
        self.processor = None
        self.vit_model = None
        
        # Trained classifier artifacts
        self.svm_pipeline = None
        self.label_encoder = None
        self.embeddings_store = None
        self.config = {}
        
        self.load_models()

    def _ensure_vit_loaded(self):
        if self.processor is None or self.vit_model is None:
            print(f"[FaceEngine] Loading Vision Transformer ({self.model_name}) on {self.device}...")
            self.processor = ViTImageProcessor.from_pretrained(self.model_name)
            self.vit_model = ViTModel.from_pretrained(self.model_name).to(self.device)
            self.vit_model.eval()
            print("[FaceEngine] Vision Transformer loaded successfully!")

    def load_models(self):
        config_path = os.path.join(MODELS_DIR, "config.json")
        pipeline_path = os.path.join(MODELS_DIR, "vit_svm_classifier.pkl")
        encoder_path = os.path.join(MODELS_DIR, "label_encoder.pkl")
        store_path = os.path.join(MODELS_DIR, "embeddings_store.pkl")

        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
                self.confidence_threshold = self.config.get("confidence_threshold", 0.65)
                self.model_name = self.config.get("model_name", self.model_name)

        if os.path.exists(pipeline_path) and os.path.exists(encoder_path):
            print("[FaceEngine] Loading pre-trained ViT+SVM deployment artifacts...")
            self.svm_pipeline = joblib.load(pipeline_path)
            self.label_encoder = joblib.load(encoder_path)
            if os.path.exists(store_path):
                self.embeddings_store = joblib.load(store_path)
            print(f"[FaceEngine] Loaded {len(self.label_encoder.classes_)} enrolled identities into classifier.")
        else:
            print("[FaceEngine] Pre-trained models not found yet. Models must be trained using train_and_export.py.")

    def detect_faces(self, frame_bgr, min_size=(60, 60)):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.15,
            minNeighbors=5,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces

    def check_face_quality(self, face_bgr):
        """
        Validates minimum resolution and sharpness (Laplacian variance).
        Returns: (is_valid: bool, reason: str, blur_score: float)
        """
        h, w = face_bgr.shape[:2]
        if h < 60 or w < 60:
            return False, f"Face crop too small ({w}x{h} px, min 60x60)", 0.0

        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < 25.0:
            return False, f"Image is blurry (sharpness score {blur_score:.1f} < 25.0)", blur_score

        return True, "Quality OK", blur_score

    def extract_embedding(self, face_bgr):
        """
        Preprocesses a cropped face and extracts 768-dim ViT CLS token embedding.
        """
        self._ensure_vit_loaded()
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (224, 224))
        pil_img = Image.fromarray(face_resized)

        inputs = self.processor(images=[pil_img], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.vit_model(**inputs)
            cls_token = outputs.last_hidden_state[:, 0, :]
            embedding = cls_token.cpu().numpy()

        return embedding # Shape (1, 768)

    def predict_identity(self, face_bgr, threshold=None):
        """
        Predicts identity with unknown-person rejection mechanism.
        Returns dict:
          - is_known: bool
          - identity: str ("Unknown" or enrolled name)
          - confidence: float (0.0 to 100.0)
          - raw_score: float (0.0 to 1.0)
          - all_probabilities: dict
        """
        if self.svm_pipeline is None or self.label_encoder is None:
            return {
                "is_known": False,
                "identity": "Model Not Ready",
                "confidence": 0.0,
                "raw_score": 0.0,
                "all_probabilities": {}
            }

        thresh = threshold if threshold is not None else self.confidence_threshold
        embedding = self.extract_embedding(face_bgr) # (1, 768)

        # Predict probabilities via SVM pipeline (StandardScaler + SVC)
        probs = self.svm_pipeline.predict_proba(embedding)[0]
        max_idx = int(np.argmax(probs))
        max_prob = float(probs[max_idx])
        predicted_name = str(self.label_encoder.inverse_transform([max_idx])[0])

        all_probs = {
            str(self.label_encoder.inverse_transform([i])[0]): round(float(p) * 100, 2)
            for i, p in enumerate(probs)
        }

        # Unknown-person rejection mechanism
        if max_prob < thresh:
            return {
                "is_known": False,
                "identity": "Unknown",
                "confidence": round(max_prob * 100, 2),
                "raw_score": max_prob,
                "threshold": thresh,
                "candidate": predicted_name,
                "all_probabilities": all_probs
            }
        else:
            return {
                "is_known": True,
                "identity": predicted_name,
                "confidence": round(max_prob * 100, 2),
                "raw_score": max_prob,
                "threshold": thresh,
                "candidate": predicted_name,
                "all_probabilities": all_probs
            }

    def enroll_student_faces(self, student_name, face_images_bgr):
        """
        Enrolls a new student dynamically:
        1. Saves face crops to dataset/face_dataset/<student_name>/
        2. Extracts ViT embeddings
        3. Updates embeddings_store.pkl
        4. Retrains the deployment SVM pipeline in < 1 second
        5. Persists updated models so recognition is immediate
        """
        self._ensure_vit_loaded()
        sanitized_name = student_name.strip().replace(" ", "_")
        student_folder = os.path.join(DATASET_DIR, sanitized_name)
        os.makedirs(student_folder, exist_ok=True)

        new_embeddings = []
        saved_count = 0

        for idx, img_bgr in enumerate(face_images_bgr, start=1):
            is_valid, reason, _ = self.check_face_quality(img_bgr)
            if not is_valid:
                continue

            save_path = os.path.join(student_folder, f"enroll_{idx:03d}.jpg")
            cv2.imwrite(save_path, img_bgr)
            saved_count += 1

            emb = self.extract_embedding(img_bgr)
            new_embeddings.append(emb)

        if len(new_embeddings) == 0:
            return False, "No valid high-quality face images passed validation."

        new_X = np.concatenate(new_embeddings, axis=0) # (N, 768)
        new_y = np.array([sanitized_name] * new_X.shape[0])

        # Merge with existing embeddings store
        store_path = os.path.join(MODELS_DIR, "embeddings_store.pkl")
        if os.path.exists(store_path):
            store = joblib.load(store_path)
            # Remove any older embeddings for this student to allow clean re-enrollment
            mask = store["labels"] != sanitized_name
            X_all = np.vstack([store["embeddings"][mask], new_X]) if np.any(mask) else new_X
            y_all = np.concatenate([store["labels"][mask], new_y]) if np.any(mask) else new_y
        else:
            X_all = new_X
            y_all = new_y

        # Refit LabelEncoder & SVM Pipeline
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y_all)

        svm_pipeline = make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=1.0, probability=True, random_state=42)
        )
        svm_pipeline.fit(X_all, y_encoded)

        # Save artifacts
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(svm_pipeline, os.path.join(MODELS_DIR, "vit_svm_classifier.pkl"))
        joblib.dump(label_encoder, os.path.join(MODELS_DIR, "label_encoder.pkl"))
        joblib.dump({"embeddings": X_all, "labels": y_all, "classes": list(label_encoder.classes_)}, store_path)

        config = {
            "model_name": self.model_name,
            "feature_dim": 768,
            "input_resolution": [224, 224],
            "confidence_threshold": self.confidence_threshold,
            "classes": list(label_encoder.classes_),
            "num_classes": len(label_encoder.classes_),
            "total_samples": int(X_all.shape[0])
        }
        with open(os.path.join(MODELS_DIR, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)

        # Update in-memory models
        self.svm_pipeline = svm_pipeline
        self.label_encoder = label_encoder
        self.config = config
        self.embeddings_store = {"embeddings": X_all, "labels": y_all}

        print(f"[FaceEngine] Successfully enrolled '{sanitized_name}' with {saved_count} face samples! Model updated.")
        return True, f"Successfully enrolled {saved_count} face images. Model updated with {len(label_encoder.classes_)} classes."

