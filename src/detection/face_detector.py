"""
Módulo para detecção facial usando InsightFace
"""

from typing import List
from config.settings import FACE_DETECTION_CONFIDENCE


def detect_faces(frame, face_analysis_model) -> List:
    """
    Detecta faces no frame usando InsightFace

    Args:
        frame: Frame de vídeo (numpy array)
        face_analysis_model: Modelo FaceAnalysis carregado

    Returns:
        Lista de objetos Face do InsightFace com score >= threshold

        Cada face contém:
        - bbox: Bounding box [x1, y1, x2, y2]
        - kps: Landmarks faciais (5 pontos: olhos, nariz, cantos da boca)
        - det_score: Score de detecção (0-1)
        - embedding: Vetor de características faciais (512-d)
        - age: Idade estimada
        - gender: Gênero (0=feminino, 1=masculino)
    """
    faces = face_analysis_model.get(frame)

    # Filtra faces com score acima do threshold
    filtered_faces = [face for face in faces if face.det_score >= FACE_DETECTION_CONFIDENCE]

    return filtered_faces
