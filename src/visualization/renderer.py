"""
Módulo para renderização de UI sobre o vídeo
"""

import cv2
from typing import List, Dict
from config.settings import SKELETON_LINKS
from visualization.colors import (
    BODY_BBOX_COLOR,
    FACE_BBOX_COLOR,
    SKELETON_COLOR,
    OBJECT_BBOX_COLOR,
    ID_COLOR,
    EMOTION_COLOR,
    ACTIVITY_COLOR,
    FACE_LABEL_COLOR,
    TEXT_COLOR,
    DEBUG_COLOR,
    ANOMALY_COLOR,
    TRACK_ID_COLOR
)


def draw_skeleton(frame, keypoints):
    """
    Desenha esqueleto (skeleton) sobre a pessoa usando keypoints

    Args:
        frame: Frame de vídeo
        keypoints: Array de keypoints (Nx3: x, y, confidence)
    """
    # Desenha conexões
    for idx1, idx2 in SKELETON_LINKS:
        if (idx1 < len(keypoints) and idx2 < len(keypoints) and
            keypoints[idx1][2] > 0.5 and keypoints[idx2][2] > 0.5):
            pt1 = (int(keypoints[idx1][0]), int(keypoints[idx1][1]))
            pt2 = (int(keypoints[idx2][0]), int(keypoints[idx2][1]))
            cv2.line(frame, pt1, pt2, SKELETON_COLOR, 2)

    # Desenha keypoints (apenas corpo, ignora cabeça)
    for i in range(5, 17):  # Índices 5-16 são corpo/membros
        if i < len(keypoints) and keypoints[i][2] > 0.5:
            cv2.circle(
                frame,
                (int(keypoints[i][0]), int(keypoints[i][1])),
                3,
                (0, 0, 255),  # Vermelho
                -1
            )


def draw_objects(frame, objects: List[Dict]):
    """
    Desenha bounding boxes dos objetos detectados

    Args:
        frame: Frame de vídeo
        objects: Lista de dicionários {'label': str, 'bbox': (x1, y1, x2, y2)}
    """
    for obj in objects:
        x1, y1, x2, y2 = obj['bbox']
        label = obj['label']

        cv2.rectangle(frame, (x1, y1), (x2, y2), OBJECT_BBOX_COLOR, 1)
        cv2.putText(
            frame, label, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            OBJECT_BBOX_COLOR, 1
        )


def draw_person_ui(frame, text_buffer: List[Dict], person_data: Dict):
    """
    Desenha UI de uma pessoa (bboxes, labels, skeleton)

    Args:
        frame: Frame de vídeo
        text_buffer: Buffer para acumular textos a serem desenhados
        person_data: Dicionário com dados da pessoa
            {
                'coords': (x1, y1, x2, y2),
                'track_id': int,
                'face_id': int,
                'face_bbox': [x1, y1, x2, y2],
                'emotion': str,
                'activity': str,
                'kps': keypoints,
                'anomalies': [str, ...]
            }
    """
    x1, y1, x2, y2 = person_data['coords']
    face_bbox = person_data['face_bbox']
    face_id = person_data['face_id']
    emotion = person_data['emotion']
    activity = person_data['activity']

    # Desenha bounding box do corpo
    cv2.rectangle(frame, (x1, y1), (x2, y2), BODY_BBOX_COLOR, 2)

    # Desenha esqueleto
    if 'kps' in person_data and person_data['kps'] is not None:
        draw_skeleton(frame, person_data['kps'])

    # Desenha bounding box da face
    if face_bbox is not None:
        fx1, fy1, fx2, fy2 = face_bbox
        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), FACE_BBOX_COLOR, 2)

    track_id = person_data.get('track_id', None)

    # Prepara labels
    labels = [
        (f"BodyTrack: T{track_id}" if track_id is not None else "BodyTrack: --", TRACK_ID_COLOR),
        (f"FaceID: {face_id}" if face_id else "FaceID: ?", ID_COLOR),
        ("Face: DETECTED" if face_id else "Face: --", FACE_LABEL_COLOR),
        (f"Emo: {emotion}" if emotion else "Emo: --", EMOTION_COLOR),
        (f"Ativ: {activity}", ACTIVITY_COLOR)
    ]

    # Adiciona debug info se disponível
    debug_info = person_data.get("pose_dbg", None)
    if debug_info:
        labels.append((debug_info, DEBUG_COLOR))

    # Adiciona anomalias
    anomalies_text = " | ".join(person_data.get("anomalies", []))
    if anomalies_text:
        labels.append((f"⚠ {anomalies_text}", ANOMALY_COLOR))

    # Desenha labels
    current_y = y1 + 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4

    for text, bg_color in labels:
        if current_y > y2:
            break

        (text_width, text_height), _ = cv2.getTextSize(text, font, scale, 1)

        # Desenha fundo com transparência simulada (cor mais clara)
        bg_alpha = tuple(int(c * 0.7 + 255 * 0.3) for c in bg_color)
        cv2.rectangle(
            frame,
            (x1 + 5, current_y - text_height - 4),
            (x1 + 5 + text_width + 8, current_y + 4),
            bg_alpha,
            -1
        )

        # Adiciona texto ao buffer (será desenhado depois)
        text_buffer.append({
            'text': text,
            'pos': (x1 + 10, current_y),
            'color': TEXT_COLOR
        })

        current_y += (text_height + 10)


def draw_anomaly_banner(frame, anomalies: List[str], frame_width: int):
    """
    Desenha banner vermelho no topo quando há anomalias

    Args:
        frame: Frame de vídeo
        anomalies: Lista de anomalias detectadas
        frame_width: Largura do frame
    """
    if not anomalies:
        return None

    # Desenha retângulo vermelho no topo
    cv2.rectangle(frame, (0, 0), (frame_width, 40), ANOMALY_COLOR, -1)

    # Texto de anomalia
    anomaly_text = f"ANOMALIA: {', '.join(set(anomalies))}"

    return {
        'text': anomaly_text,
        'pos': (10, 28),
        'color': (255, 255, 255)  # Branco
    }


def draw_text_buffer(frame, text_buffer: List[Dict]):
    """
    Desenha todos os textos acumulados no buffer

    Args:
        frame: Frame de vídeo
        text_buffer: Lista de dicionários {'text': str, 'pos': (x, y), 'color': (b, g, r)}
    """
    for text_item in text_buffer:
        # Usa font scale maior para anomalias
        font_scale = 0.8 if 'ANOMALIA' in text_item['text'] else 0.4

        cv2.putText(
            frame,
            text_item['text'],
            text_item['pos'],
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_item['color'],
            1,
            cv2.LINE_AA
        )


def render_frame(frame, render_data, anomalies_list, objects, should_process_ai, frame_width):
    """
    Renderiza todos os elementos visuais no frame.
    Orquestra as funções de desenho existentes.

    Args:
        frame: Frame de vídeo
        render_data: Lista de pessoas detectadas
        anomalies_list: Lista de anomalias
        objects: Lista de objetos detectados
        should_process_ai: Se deve desenhar objetos
        frame_width: Largura do frame

    Returns:
        frame anotado (modifica in-place, mas retorna para clareza)
    """
    # Desenha objetos detectados
    if should_process_ai:
        draw_objects(frame, objects)

    text_buffer = []

    # Banner de anomalia
    if anomalies_list:
        anomaly_text = draw_anomaly_banner(frame, anomalies_list, frame_width)
        if anomaly_text:
            text_buffer.append(anomaly_text)

    # Desenha UI de cada pessoa
    for item in render_data:
        draw_person_ui(frame, text_buffer, item)

    # Desenha todos os textos
    draw_text_buffer(frame, text_buffer)

    return frame
