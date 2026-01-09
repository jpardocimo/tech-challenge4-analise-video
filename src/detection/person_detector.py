"""
Módulo para detecção de pessoas com tracking usando YOLO Pose
"""

from typing import List, Tuple, Optional
import numpy as np
from config.settings import PERSON_DETECTION_CONFIDENCE, PERSON_TRACKING_IOU


def detect_people_with_tracking(frame, yolo_pose_model) -> List[Tuple]:
    """
    Detecta pessoas no frame com tracking persistente usando ByteTrack

    Args:
        frame: Frame de vídeo (numpy array)
        yolo_pose_model: Modelo YOLO Pose carregado

    Returns:
        Lista de tuplas com formato:
        [(x1, y1, x2, y2, confidence, keypoints, track_id), ...]

        Onde:
        - x1, y1, x2, y2: Bounding box da pessoa
        - confidence: Confiança da detecção (0-1)
        - keypoints: Array Nx3 com keypoints do corpo (x, y, confidence)
        - track_id: ID único de tracking da pessoa (ou None se não disponível)
    """
    results = yolo_pose_model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],  # Classe 0 = person
        conf=PERSON_DETECTION_CONFIDENCE,
        iou=PERSON_TRACKING_IOU,
        verbose=False
    )

    bodies = []

    if not results:
        return bodies

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return bodies

    boxes = result.boxes
    track_ids = boxes.id
    keypoints = result.keypoints

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        confidence = float(box.conf[0].cpu().numpy())

        # Extrai keypoints se disponíveis
        keypoints_array = None
        if keypoints is not None:
            keypoints_array = keypoints[i].data[0]

        # Extrai track_id se disponível
        track_id = None
        if track_ids is not None:
            track_id = int(track_ids[i].cpu().numpy())

        bodies.append((x1, y1, x2, y2, confidence, keypoints_array, track_id))

    return bodies
