"""
Módulo para análise de emoções usando DeepFace
"""

from typing import Tuple, List, Dict, Optional
from deepface import DeepFace
from config.settings import (
    EMOTION_CACHE_INTERVAL,
    FACE_CROP_PADDING,
    MIN_FACE_SIZE,
    EMOTION_HISTORY_WINDOW
)


def process_emotion(
    frame,
    face,
    face_id: int,
    emotion_cache: Dict,
    frame_count: int,
    stats: Dict
) -> Tuple[Optional[str], List]:
    """
    Processa emoção de uma face usando DeepFace com cache otimizado

    Args:
        frame: Frame de vídeo completo
        face: Objeto Face do InsightFace
        face_id: ID da face
        emotion_cache: Cache de emoções {face_id: {'val': emotion, 'frame': int, 'history': []}}
        frame_count: Número do frame atual
        stats: Dicionário de estatísticas globais

    Returns:
        Tupla (emotion, emotion_history)
        - emotion: String com emoção detectada ou None
        - emotion_history: Lista de tuplas (frame, emotion) com histórico
    """
    cached = emotion_cache.get(face_id, {})
    emotion = cached.get('val', None)
    last_update_frame = cached.get('frame', 0)
    emotion_history = cached.get('history', [])

    # Verifica se deve analisar emoção (cache interval)
    should_analyze = (emotion is None or (frame_count - last_update_frame > EMOTION_CACHE_INTERVAL))

    if should_analyze:
        try:
            # Crop da face com padding
            keypoints = face.kps
            if keypoints is not None:
                height, width = frame.shape[:2]
                fx1, fy1, fx2, fy2 = face.bbox.astype(int)

                # Adiciona padding de 50%
                pad_width = int((fx2 - fx1) * FACE_CROP_PADDING)
                pad_height = int((fy2 - fy1) * FACE_CROP_PADDING)

                fx1 = max(0, fx1 - pad_width)
                fy1 = max(0, fy1 - pad_height)
                fx2 = min(width, fx2 + pad_width)
                fy2 = min(height, fy2 + pad_height)

                face_crop = frame[fy1:fy2, fx1:fx2]

                # Valida tamanho mínimo do crop
                if (face_crop.size > 0 and
                    face_crop.shape[0] > MIN_FACE_SIZE and
                    face_crop.shape[1] > MIN_FACE_SIZE):

                    # Usa DeepFace com detector_backend='skip' (já temos o crop)
                    # Isso é MUITO mais rápido que usar detecção de face do DeepFace
                    result = DeepFace.analyze(
                        face_crop,
                        actions=['emotion'],
                        enforce_detection=False,
                        detector_backend='skip',
                        silent=True
                    )

                    new_emotion = result[0]['dominant_emotion']

                    if new_emotion:
                        emotion = new_emotion
                        stats['emocoes'][emotion] += 1

                        # Adiciona ao histórico se mudou
                        if not emotion_history or emotion_history[-1][1] != emotion:
                            emotion_history.append((frame_count, emotion))

                        # Mantém apenas últimos N frames no histórico
                        emotion_history = [
                            entry for entry in emotion_history
                            if frame_count - entry[0] < EMOTION_HISTORY_WINDOW
                        ]
                else:
                    emotion = "neutral"

        except Exception as e:
            # Fallback para neutral em caso de erro
            emotion = "neutral" if emotion is None else emotion

        # Atualiza cache
        emotion_cache[face_id] = {
            'val': emotion,
            'frame': frame_count,
            'history': emotion_history
        }

    return emotion, emotion_history
