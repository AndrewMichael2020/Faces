# Face Embedding & Similarity Pipeline

This project demonstrates how to use MTCNN (from facenet-pytorch) for face detection and InsightFace for face embedding extraction, enabling robust face similarity comparison. The code is written in Python and is compatible with CPU-only environments.

## Features
- **Face Detection:** Uses MTCNN to detect and crop faces from images.
- **Face Embedding:** Uses InsightFace's ArcFace model to extract embeddings from cropped faces.
- **Similarity Scoring:** Computes cosine similarity between two face embeddings.

## Usage
1. Place your images (e.g., `person1.jpg`, `person2.jpg`) in the project directory.
2. Run the script:
   ```bash
   python3 Faces.py
   ```
3. The script will print the similarity score between the two faces.

## Code Overview
- **MTCNN** is used for accurate face detection and cropping.
- **InsightFace** is used for embedding extraction from the cropped face (resized to 112x112, BGR format).
- **Cosine Similarity** is used to compare the embeddings.

## Options & Improvements

### 1. Multi-Embedding Matching
- **Multiple Faces per Image:**
  - Modify the code to detect and embed all faces in an image (set `keep_all=True` in MTCNN).
  - Compare all pairs of embeddings between two images (e.g., using a distance matrix).
- **Best Match:**
  - For each face in image A, find the most similar face in image B.
  - Use the maximum, minimum, or average similarity as your metric.

### 2. Batch Processing
- Process multiple images in a directory and build a similarity matrix for all pairs.
- Useful for clustering or face search applications.

### 3. Embedding Storage
- Save embeddings to disk (e.g., as `.npy` files) for fast future comparisons.
- Build an index for large-scale face search (e.g., using FAISS).

### 4. Model Options
- Try different InsightFace models (e.g., `antelopev2`, `glintr100`) for different trade-offs in speed and accuracy.
- Use GPU if available for faster processing.

### 5. Preprocessing
- Improve alignment by using facial landmarks.
- Apply histogram equalization or other normalization for challenging lighting conditions.

### 6. Thresholding
- Set a similarity threshold to decide if two faces are the same person.
- Calibrate this threshold on your own dataset for best results.

## Example: Multi-Embedding Matching
```python
# Detect all faces in both images
mtcnn = MTCNN(keep_all=True)
faces1 = mtcnn(Image.open('person1.jpg').convert('RGB'))
faces2 = mtcnn(Image.open('person2.jpg').convert('RGB'))

# Extract embeddings for all faces
embeddings1 = [extract_embedding(face) for face in faces1]
embeddings2 = [extract_embedding(face) for face in faces2]

# Compute similarity matrix
similarity_matrix = cosine_similarity(embeddings1, embeddings2)

# Find best matches
best_matches = similarity_matrix.max(axis=1)
```

## Requirements
- Python 3.8+
- facenet-pytorch
- insightface
- numpy
- scikit-learn
- pillow
- opencv-python

Install requirements:
```bash
pip install facenet-pytorch insightface numpy scikit-learn pillow opencv-python
```

## References
- [InsightFace](https://github.com/deepinsight/insightface)
- [facenet-pytorch](https://github.com/timesler/facenet-pytorch)

---
Feel free to extend this project for your own face recognition, clustering, or search applications!
