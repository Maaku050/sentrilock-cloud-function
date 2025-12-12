"""
InsightFace Recognition Service
Install: pip install insightface onnxruntime flask flask-cors pillow numpy
Run: python insightface_service.py

This uses the Buffalo_L model which provides excellent accuracy and speed.
"""

from insightface.app import FaceAnalysis
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
import cv2

app = Flask(__name__)
CORS(app)

# Global face analysis app
face_app = None

def initialize_face_app():
    """Initialize InsightFace app with Buffalo_L model"""
    global face_app
    if face_app is not None:
        return
    
    print("="*60)
    print("Initializing InsightFace...")
    print("="*60)
    
    # Initialize with Buffalo_L model (best balance of speed/accuracy)
    # Models will be downloaded automatically on first run
    face_app = FaceAnalysis(
        name='buffalo_l',  # or 'buffalo_s' for faster, 'buffalo_sc' for smallest
        providers=['CPUExecutionProvider']  # Use 'CUDAExecutionProvider' if you have GPU
    )
    
    # Prepare for detection with input size 640x640
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    
    print("✓ InsightFace initialized successfully")
    print("="*60)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    is_ready = face_app is not None
    return jsonify({
        'status': 'healthy' if is_ready else 'initializing',
        'service': 'InsightFace Recognition Service',
        'model': 'buffalo_l',
        'ready': is_ready
    })

@app.route('/extract-embedding', methods=['POST'])
def extract_embedding():
    """
    Extract 512-dimensional face embedding from image
    Returns embedding, bounding box, landmarks, and other face attributes
    """
    try:
        # Initialize if not already done
        initialize_face_app()
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        image_bytes = image_file.read()
        
        # Load image with PIL then convert to numpy/cv2 format
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array (OpenCV format: BGR)
        image_array = np.array(image)
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Detect faces and extract features
        faces = face_app.get(image_bgr)
        
        if len(faces) == 0:
            return jsonify({
                'success': False,
                'message': 'No face detected in image'
            }), 200
        
        # Use the first (most prominent) face
        face = faces[0]
        
        # Extract embedding (512-dimensional vector)
        embedding = face.embedding.tolist()
        
        # Get bounding box
        bbox = face.bbox.astype(int).tolist()
        
        # Get landmarks (5 points: left_eye, right_eye, nose, left_mouth, right_mouth)
        landmarks = face.kps.astype(int).tolist() if hasattr(face, 'kps') else []
        
        # Get additional attributes if available
        gender = None
        age = None
        
        if hasattr(face, 'gender'):
            gender = 'male' if face.gender == 1 else 'female'
        
        if hasattr(face, 'age'):
            age = int(face.age)
        
        # Calculate face quality score (based on detection score)
        quality_score = float(face.det_score) if hasattr(face, 'det_score') else 1.0
        
        response = {
            'success': True,
            'embedding': embedding,
            'embedding_size': len(embedding),
            'num_faces_detected': len(faces),
            'face': {
                'bbox': {
                    'x1': bbox[0],
                    'y1': bbox[1],
                    'x2': bbox[2],
                    'y2': bbox[3]
                },
                'landmarks': landmarks,
                'quality_score': quality_score,
                'gender': gender,
                'age': age
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/compare-embeddings', methods=['POST'])
def compare_embeddings():
    """
    Compare two face embeddings using cosine similarity
    InsightFace embeddings are already normalized, so we can use dot product
    """
    try:
        data = request.json
        
        if not data or 'embedding1' not in data or 'embedding2' not in data:
            return jsonify({'error': 'Two embeddings required'}), 400
        
        emb1 = np.array(data['embedding1'], dtype=np.float32)
        emb2 = np.array(data['embedding2'], dtype=np.float32)
        
        # Normalize embeddings (InsightFace embeddings should already be normalized)
        emb1_norm = emb1 / np.linalg.norm(emb1)
        emb2_norm = emb2 / np.linalg.norm(emb2)
        
        # Calculate cosine similarity (dot product of normalized vectors)
        similarity = np.dot(emb1_norm, emb2_norm)
        
        # Convert similarity to distance (0 = identical, 1 = completely different)
        distance = (1 - similarity) / 2
        
        # Convert to percentage
        similarity_percent = float(similarity * 100)
        
        # Threshold for matching (typical: 0.25-0.35 for InsightFace)
        # Lower distance = better match
        is_match = distance < 0.30
        
        # Confidence level
        if distance < 0.20:
            confidence = 'very_high'
        elif distance < 0.30:
            confidence = 'high'
        elif distance < 0.40:
            confidence = 'medium'
        else:
            confidence = 'low'
        
        return jsonify({
            'success': True,
            'distance': float(distance),
            'similarity': similarity_percent,
            'is_match': is_match,
            'confidence': confidence,
            'threshold_used': 0.30
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/batch-compare', methods=['POST'])
def batch_compare():
    """
    Compare one embedding against multiple stored embeddings
    Returns all matches above threshold, sorted by similarity
    """
    try:
        data = request.json
        
        if not data or 'query_embedding' not in data or 'stored_embeddings' not in data:
            return jsonify({'error': 'query_embedding and stored_embeddings required'}), 400
        
        query_emb = np.array(data['query_embedding'], dtype=np.float32)
        stored_embs = data['stored_embeddings']  # List of {id, name, embedding}
        
        # Normalize query embedding
        query_emb_norm = query_emb / np.linalg.norm(query_emb)
        
        matches = []
        
        for item in stored_embs:
            stored_emb = np.array(item['embedding'], dtype=np.float32)
            stored_emb_norm = stored_emb / np.linalg.norm(stored_emb)
            
            # Calculate similarity
            similarity = np.dot(query_emb_norm, stored_emb_norm)
            distance = (1 - similarity) / 2
            
            if distance < 0.40:  # Only return reasonable matches
                matches.append({
                    'id': item.get('id'),
                    'name': item.get('name'),
                    'distance': float(distance),
                    'similarity': float(similarity * 100),
                    'is_match': distance < 0.30
                })
        
        # Sort by distance (ascending - best matches first)
        matches.sort(key=lambda x: x['distance'])
        
        return jsonify({
            'success': True,
            'matches': matches,
            'total_compared': len(stored_embs),
            'matches_found': len(matches)
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/analyze-face', methods=['POST'])
def analyze_face():
    """
    Full face analysis: detection, embedding, gender, age
    """
    try:
        initialize_face_app()
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        image_file = request.files['image']
        image_bytes = image_file.read()
        
        # Process image
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_array = np.array(image)
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        
        # Analyze all faces
        faces = face_app.get(image_bgr)
        
        if len(faces) == 0:
            return jsonify({
                'success': False,
                'message': 'No faces detected'
            }), 200
        
        results = []
        for idx, face in enumerate(faces):
            face_data = {
                'face_id': idx,
                'embedding': face.embedding.tolist(),
                'bbox': face.bbox.astype(int).tolist(),
                'quality_score': float(face.det_score) if hasattr(face, 'det_score') else None,
                'gender': 'male' if hasattr(face, 'gender') and face.gender == 1 else 'female' if hasattr(face, 'gender') else None,
                'age': int(face.age) if hasattr(face, 'age') else None
            }
            results.append(face_data)
        
        return jsonify({
            'success': True,
            'num_faces': len(faces),
            'faces': results
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("="*60)
    print("InsightFace Recognition Service")
    print("="*60)
    print("Starting server...")
    print("Note: Models will be downloaded on first request (~100MB)")
    print("="*60)
    print("\nEndpoints:")
    print("  GET  /health              - Health check")
    print("  POST /extract-embedding   - Extract face embedding")
    print("  POST /compare-embeddings  - Compare two embeddings")
    print("  POST /batch-compare       - Compare one vs many embeddings")
    print("  POST /analyze-face        - Full face analysis")
    print("="*60)
    
    # Pre-initialize the model
    try:
        initialize_face_app()
    except Exception as e:
        print(f"Warning: Could not pre-initialize: {e}")
        print("Model will be loaded on first request")
    
    app.run(host='0.0.0.0', port=5000, debug=True)