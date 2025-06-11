from facenet_pytorch import MTCNN
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from PIL import Image
import cv2

# Init MTCNN and InsightFace
mtcnn = MTCNN(keep_all=False)
insight = FaceAnalysis(name='buffalo_l')
insight.prepare(ctx_id=-1)

def detect_and_embed(img_path):
    img = Image.open(img_path).convert("RGB")
    face = mtcnn(img)
    if face is None:
        raise ValueError(f"No face found in {img_path}")
    face_np = (face.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    face_np = cv2.resize(face_np, (112, 112))
    face_np = cv2.cvtColor(face_np, cv2.COLOR_RGB2BGR)
    recog = insight.models['recognition']
    if hasattr(recog, 'get_embedding'):
        embedding = recog.get_embedding(face_np)
    elif hasattr(recog, 'compute_embedding'):
        embedding = recog.compute_embedding(face_np)
    elif hasattr(recog, 'get_feat'):
        embedding = recog.get_feat(face_np)
    else:
        raise RuntimeError('Recognition model does not have get_embedding, compute_embedding, or get_feat method')
    if embedding is None:
        raise ValueError(f"No embedding found in {img_path}")
    return embedding

# Load embeddings
embedding1 = detect_and_embed("person1.jpg").reshape(1, -1)
embedding2 = detect_and_embed("person2.jpg").reshape(1, -1)

# Similarity score
score = cosine_similarity(embedding1, embedding2)[0][0]
print(f"Similarity: {score:.4f}")
