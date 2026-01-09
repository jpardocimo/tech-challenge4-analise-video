"""
Módulo para carregamento centralizado de modelos de IA
"""

from ultralytics import YOLO
from insightface.app import FaceAnalysis
from config.settings import (
    YOLO_POSE_MODEL_PATH,
    YOLO_OBJECT_MODEL_PATH,
    INSIGHTFACE_MODEL_NAME
)


def load_models():
    """
    Carrega todos os modelos de IA necessários

    Returns:
        Tupla (yolo_pose_model, yolo_object_model, face_analysis_model)
        - yolo_pose_model: YOLO para detecção de pose
        - yolo_object_model: YOLO para detecção de objetos
        - face_analysis_model: InsightFace para detecção e análise facial
    """
    print("📦 Carregando modelos...")

    # Carrega YOLO Pose
    print("  - Carregando YOLO Pose...")
    yolo_pose_model = YOLO(YOLO_POSE_MODEL_PATH)

    # Carrega YOLO Object Detection
    print("  - Carregando YOLO Object Detection...")
    yolo_object_model = YOLO(YOLO_OBJECT_MODEL_PATH)

    # Carrega InsightFace
    print("  - Carregando InsightFace...")
    face_analysis_model = FaceAnalysis(
        name=INSIGHTFACE_MODEL_NAME,
        providers=['CPUExecutionProvider']
    )
    face_analysis_model.prepare(ctx_id=0, det_size=(640, 640))

    # DeepFace não precisa de carregamento prévio - lazy loading
    print("  - DeepFace configurado para análise de emoções (lazy loading)")

    print("✅ Todos os modelos carregados com sucesso!")

    return yolo_pose_model, yolo_object_model, face_analysis_model
