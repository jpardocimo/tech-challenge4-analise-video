"""
Módulo para associação entre faces detectadas e corpos

Responsabilidades:
- Associar faces aos corpos detectados (matching espacial)
- Orquestrar análise completa (pose + emoção + atividade)
- Gerar dados consolidados para renderização
"""

from typing import List, Dict, Tuple
from config.settings import (
    POSTURE_LABELS,
    EMOTION_INSTABILITY_THRESHOLD,
    SHOW_POSTURE_IN_ACTIVITY_LABEL,
    DEFAULT_ACTIVITY_LABEL,
    POSTURE_ACTIVITY_LABELS,
    ANOMALY_MESSAGES
)
from core.pose_analysis import analyze_body_pose
from models.face_identity import resolve_face_identity


def match_faces_to_bodies(
    frame,
    bodies: List[Tuple],
    faces: List,
    frame_count: int,
    detected_objects: List[Dict],
    activity_classifier,
    state_manager,
    stats_manager,
    emotion_manager
) -> Tuple[List[Dict], List[str]]:
    """
    Associa faces detectadas aos corpos e processa identidade, emoção e atividade

    Args:
        frame: Frame de vídeo atual
        bodies: Lista de tuplas (x1, y1, x2, y2, conf, keypoints, track_id)
        faces: Lista de faces detectadas (InsightFace)
        frame_count: Número do frame atual
        detected_objects: Lista de objetos detectados
        activity_classifier: Instância do ActivityClassifier
        state_manager: Instância do StateManager
        stats_manager: Instância do StatsManager
        emotion_manager: Instância do EmotionManager

    Returns:
        Tupla (render_data, anomalies_list)
        - render_data: Lista de dicionários com dados para renderização
        - anomalies_list: Lista de strings com anomalias detectadas
    """
    render_data = []
    unmatched_faces_indices = list(range(len(faces)))
    frame_anomalies = []

    for body_index, (bx1, by1, bx2, by2, body_conf, keypoints, track_id) in enumerate(bodies):
        # Verifica objetos próximos ao corpo
        nearby_object_labels = []
        for obj in detected_objects:
            ox1, oy1, ox2, oy2 = obj['bbox']
            object_center_x = (ox1 + ox2) / 2
            object_center_y = (oy1 + oy2) / 2

            # Verifica se objeto está próximo (100px de margem)
            if (bx1 - 100 < object_center_x < bx2 + 100) and (by1 - 100 < object_center_y < by2 + 100):
                nearby_object_labels.append(obj['label'])

        body_data = {
            'coords': (bx1, by1, bx2, by2),
            'track_id': track_id,
            'activity': DEFAULT_ACTIVITY_LABEL,  # Usa constante do settings
            'face_id': None,
            'face_bbox': None,
            'emotion': None,
            'anomalies': [],
            'kps': keypoints,
            'objects': nearby_object_labels
        }

        # Análise de pose
        pose_analysis = analyze_body_pose(keypoints, conf_min=0.6)
        body_data['pose_analysis'] = pose_analysis
        body_data['posture'] = pose_analysis['posture']
        body_data['body_pose_confidence'] = pose_analysis['confidence']

        # Define atividade baseada em postura (apenas se flag estiver ativa)
        # Caso contrário, mantém DEFAULT_ACTIVITY_LABEL
        if not SHOW_POSTURE_IN_ACTIVITY_LABEL:
            # Não mostra postura, mantém fallback padrão
            pass
        else:
            # Mostra postura como atividade fallback
            posture_label = POSTURE_ACTIVITY_LABELS.get(pose_analysis['posture'])
            if posture_label:
                body_data['activity'] = posture_label

        # Detecção de queda brusca (mudança standing -> lying_down)
        temp_track_id = track_id if track_id is not None else (-1000 - body_index)
        action_state = state_manager.get_action_state(temp_track_id)

        if action_state is None:
            from core.activity_analysis import ActionState
            action_state = ActionState()
            state_manager.set_action_state(temp_track_id, action_state)

        current_posture = pose_analysis['posture']
        if action_state.previous_posture is not None:
            # Detecta queda: estava em pé/sentado e agora está deitado
            if action_state.previous_posture in ['standing', 'sitting'] and current_posture == 'lying_down':
                body_data['anomalies'].append(ANOMALY_MESSAGES['fall_detected'])  # Usa constante
                stats_manager.add_anomalia()

        action_state.previous_posture = current_posture

        # Encontra face dentro do corpo
        faces_in_body = []
        for face_index in unmatched_faces_indices:
            face = faces[face_index]
            fx1, fy1, fx2, fy2 = face.bbox.astype(int)
            face_center_x = fx1 + (fx2 - fx1) / 2
            face_center_y = fy1 + (fy2 - fy1) / 2

            if (face_center_x > bx1 and face_center_x < bx2 and
                face_center_y > by1 and face_center_y < by2):
                faces_in_body.append(face_index)

        # Seleciona a maior face
        best_face_index = None
        if faces_in_body:
            best_face_index = max(
                faces_in_body,
                key=lambda i: (faces[i].bbox[2] - faces[i].bbox[0]) * (faces[i].bbox[3] - faces[i].bbox[1])
            )

        face = None
        if best_face_index is not None:
            unmatched_faces_indices.remove(best_face_index)
            face = faces[best_face_index]

            # Identifica pessoa
            face_id = resolve_face_identity(face.embedding)
            stats_manager.add_pessoa(face_id)

            # Processa emoção
            emotion, emotion_history = emotion_manager.process_emotion(
                frame, face, face_id, frame_count, stats_manager
            )

            # Detecta instabilidade emocional
            if len(emotion_history) > EMOTION_INSTABILITY_THRESHOLD:
                body_data['anomalies'].append("Possivel CARETA ou Instab. Emocional")
                stats_manager.add_anomalia()

            body_data['face_id'] = face_id
            body_data['face_bbox'] = face.bbox.astype(int)
            body_data['emotion'] = emotion

        # Classifica atividade usando rule engine
        face_landmarks = face.kps if face is not None else None
        activity_label = activity_classifier.evaluate(
            keypoints,
            nearby_object_labels,
            action_state,
            face.bbox if face else None,
            face_landmarks
        )

        # Mantém activity_raw e posture separados
        body_data['activity_raw'] = activity_label

        # Cria label composta combinando postura + atividade (se flag ativa)
        if activity_label != DEFAULT_ACTIVITY_LABEL:
            # Atividade específica foi detectada
            if SHOW_POSTURE_IN_ACTIVITY_LABEL:
                # Combina postura + atividade
                posture = pose_analysis['posture']
                posture_pt = POSTURE_LABELS.get(posture, '')
                if posture_pt:
                    body_data['activity'] = f"{posture_pt} - {activity_label}"
                else:
                    body_data['activity'] = activity_label
            else:
                # Mostra apenas a atividade
                body_data['activity'] = activity_label
        # Se não detectou atividade, mantém o que já foi definido (postura ou DEFAULT_ACTIVITY_LABEL)


        # Acumula anomalias
        if body_data['anomalies']:
            frame_anomalies.extend(body_data['anomalies'])

        # Atualiza estatísticas de atividades
        stats_manager.add_atividade(body_data['activity'])

        render_data.append(body_data)

    return render_data, frame_anomalies
