"""
Módulo para classificação de atividades usando rule engine configurável

Responsabilidades:
- Classificar atividades baseado em regras configuráveis (JSON)
- Detectar ações específicas (acenar, mão no rosto, etc)
- Manter estado temporal de ações para suavização
"""

import json
import numpy as np
from typing import Dict, List, Optional, Any
from collections import deque
from config.settings import (
    ACTION_HOLD_FRAMES,
    ACTION_CONFIRM_FRAMES,
    HAND_POSITION_HISTORY_WINDOW,
    HAND_NEAR_FACE_MIN_FRAMES
)
from core.pose_analysis import get_keypoint


class ActionState:
    """
    Estado para suavização e histórico de atividades

    Mantém:
    - Atividade atual e candidata
    - Histórico de posições das mãos (para detecção de acenar)
    - Keypoints suavizados (EMA)
    - Postura anterior (para detectar quedas)
    - Contador de mão perto do rosto
    """

    def __init__(self):
        self.current = "Atividade nao detectada"
        self.candidate = None
        self.candidate_count = 0
        self.hold_until = 0

        # Histórico apenas para detecção de acenar (posição X das mãos)
        self.left_wrist_x_history = deque(maxlen=HAND_POSITION_HISTORY_WINDOW)
        self.right_wrist_x_history = deque(maxlen=HAND_POSITION_HISTORY_WINDOW)
        self.smoothed_keypoints = None

        # Histórico de postura para detectar mudanças bruscas (ex: queda)
        self.previous_posture = None

        # Contador para temporal smoothing de mão no rosto
        self.hand_near_face_count = 0


def smooth_keypoints_ema(previous, current, alpha: float = 0.6):
    """
    Suavização exponencial para reduzir jitter de keypoints

    Args:
        previous: Keypoints do frame anterior
        current: Keypoints do frame atual
        alpha: Peso do frame anterior (0-1)

    Returns:
        Keypoints suavizados
    """
    if previous is None:
        return current
    if current is None:
        return previous

    return previous * alpha + current * (1 - alpha)


class ActivityClassifier:
    """
    Engine baseada em regras para classificação de atividades

    Carrega configuração de atividades de arquivo JSON e avalia
    regras para detectar atividades baseadas em:
    - Objetos próximos
    - Condições de pose (postura, posição de braços/mãos, movimento)
    """

    def __init__(self, config_path: str):
        """
        Args:
            config_path: Caminho para arquivo JSON de configuração
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f).get('actions', {})
        except Exception as e:
            print(f"❌ Erro ao carregar actions_config: {e}")
            self.config = {}

    def evaluate(
        self,
        keypoints,
        nearby_objects: List[str],
        action_state: Optional[ActionState] = None,
        face_bbox = None,
        face_landmarks = None
    ) -> str:
        """
        Avalia regras e retorna a melhor atividade detectada

        Args:
            keypoints: Keypoints do YOLO Pose (17 pontos)
            nearby_objects: Lista de objetos detectados próximos
            action_state: Estado de ação para tracking temporal
            face_bbox: Bounding box da face
            face_landmarks: Landmarks faciais do InsightFace (5 pontos: olhos, nariz, boca)

        Returns:
            Nome da atividade detectada
        """
        best_action = "Atividade nao detectada"
        best_priority = 0

        # Suavização de keypoints
        if action_state and keypoints is not None:
            if action_state.smoothed_keypoints is None:
                action_state.smoothed_keypoints = np.array(keypoints, dtype=np.float32)
            else:
                current = np.array(keypoints, dtype=np.float32)
                if current.shape == action_state.smoothed_keypoints.shape:
                    action_state.smoothed_keypoints = smooth_keypoints_ema(
                        action_state.smoothed_keypoints,
                        current,
                        alpha=0.6
                    )
                else:
                    action_state.smoothed_keypoints = current

            keypoints = action_state.smoothed_keypoints
            self._update_hand_position_history(action_state, keypoints)

        # Análise de pose
        pose_info = self._analyze_pose(keypoints, action_state, face_bbox, face_landmarks)

        # Avalia cada ação configurada
        for action_name, rules in self.config.items():
            # Verifica objetos próximos
            required_objects = rules.get('required_objects', [])
            if required_objects:
                has_object = any(req in nearby_objects for req in required_objects)
                if not has_object:
                    continue

            # Verifica condições de pose
            pose_conditions = rules.get('pose_conditions', {})
            if not self._check_pose_conditions(pose_info, pose_conditions):
                continue

            # Verifica prioridade
            priority = rules.get('priority', 0)
            if priority > best_priority:
                best_priority = priority
                best_action = action_name.replace("_", " ").title()

        return best_action

    def _update_hand_position_history(self, state: ActionState, keypoints):
        """
        Atualiza histórico de posições das mãos para análise temporal

        Args:
            state: ActionState
            keypoints: Keypoints atuais
        """
        left_wrist = get_keypoint(keypoints, 'left_wrist', conf_min=0.4)
        right_wrist = get_keypoint(keypoints, 'right_wrist', conf_min=0.4)

        # Histórico das posições X das mãos (para detectar acenar)
        if left_wrist is not None:
            state.left_wrist_x_history.append(float(left_wrist[0]))

        if right_wrist is not None:
            state.right_wrist_x_history.append(float(right_wrist[0]))

    def _analyze_pose(
        self,
        keypoints,
        state: Optional[ActionState],
        face_bbox,
        face_landmarks
    ) -> Dict:
        """
        Analisa pose e retorna informações estruturadas

        Args:
            keypoints: Keypoints do YOLO Pose (17 pontos do corpo)
            state: ActionState para tracking temporal
            face_bbox: Bounding box da face
            face_landmarks: Landmarks faciais do InsightFace (5 pontos)

        Returns:
            Dicionário com informações de pose:
            {
                'both_arms_up': bool,
                'hands_position': str ('on_face', 'on_forehead', 'close_to_torso', 'normal'),
                'posture': str,
                'one_hand_above_neck': bool,
                'wave_motion': bool,
                ...
            }
        """
        info = {
            'both_arms_up': False,
            'hands_on_face': False,
            'hands_on_forehead': False,
            'head_pitch': 'normal',
            'posture': 'standing',
            'legs_motion': 'standing',
            'displacement': False,
            'one_hand_above_neck': False,
            'wave_motion': False,
            'hands_position': 'normal',
            'body_angle': 'vertical',
            'debug_scores': {}
        }

        if keypoints is None:
            return info

        # Extrai keypoints principais
        nose = get_keypoint(keypoints, 'nose', conf_min=0.4)
        left_shoulder = get_keypoint(keypoints, 'left_shoulder', conf_min=0.4)
        right_shoulder = get_keypoint(keypoints, 'right_shoulder', conf_min=0.4)
        left_wrist = get_keypoint(keypoints, 'left_wrist', conf_min=0.4)
        right_wrist = get_keypoint(keypoints, 'right_wrist', conf_min=0.4)
        left_hip = get_keypoint(keypoints, 'left_hip', conf_min=0.4)
        right_hip = get_keypoint(keypoints, 'right_hip', conf_min=0.4)

        # Detecta se pessoa está deitada (corpo horizontal)
        is_lying_down = False
        if left_shoulder is not None and right_shoulder is not None and left_hip is not None and right_hip is not None:
            shoulder_center = (left_shoulder + right_shoulder) / 2.0
            hip_center = (left_hip + right_hip) / 2.0

            dx = abs(hip_center[0] - shoulder_center[0])
            dy = abs(hip_center[1] - shoulder_center[1])

            if dx > dy * 1.2:
                is_lying_down = True
                info['posture'] = 'lying_down'
                info['body_angle'] = 'horizontal'
                return info

        # Referências
        neck_y = 0
        shoulder_width = 1.0
        if left_shoulder is not None and right_shoulder is not None:
            neck_y = (left_shoulder[1] + right_shoulder[1]) / 2.0
            shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)

        # Ambos braços levantados
        if left_wrist is not None and right_wrist is not None and nose is not None:
            if left_wrist[1] < nose[1] and right_wrist[1] < nose[1]:
                info['both_arms_up'] = True

        # Pitch da cabeça (simplificado)
        info['head_pitch'] = 'down'

        # Variância de movimento das mãos (usado apenas para "acenar")
        variance_left = 0
        variance_right = 0
        if state and len(state.left_wrist_x_history) >= 5:
            variance_left = np.var(list(state.left_wrist_x_history)[-10:]) if len(state.left_wrist_x_history) > 0 else 0
            variance_right = np.var(list(state.right_wrist_x_history)[-10:]) if len(state.right_wrist_x_history) > 0 else 0

        # Detecção de mão no rosto
        face_center = None
        nose_landmark = None

        # PRIORIDADE 1: Usa landmarks do InsightFace
        if face_landmarks is not None and len(face_landmarks) >= 5:
            if not isinstance(face_landmarks, np.ndarray):
                face_landmarks = np.array(face_landmarks, dtype=np.float32)

            face_center = np.mean(face_landmarks, axis=0)
            nose_landmark = face_landmarks[2]

        # FALLBACK: Usa keypoints do YOLO Pose
        elif left_shoulder is not None and right_shoulder is not None:
            face_points = []
            for part in ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear']:
                pt = get_keypoint(keypoints, part, conf_min=0.25)
                if pt is not None:
                    face_points.append(pt)

            if face_points:
                face_center = np.mean(face_points, axis=0)
                nose_landmark = get_keypoint(keypoints, 'nose', conf_min=0.25)

        # Processa detecção de mão no rosto
        if (face_center is not None and
            left_shoulder is not None and
            right_shoulder is not None and
            (left_wrist is not None or right_wrist is not None)):

            VERY_CLOSE_THRESHOLD = shoulder_width * 0.45

            hand_is_near_face = False
            hand_on_forehead = False

            if left_wrist is not None:
                dist = np.linalg.norm(left_wrist - face_center)
                if dist < VERY_CLOSE_THRESHOLD:
                    hand_is_near_face = True
                    if nose_landmark is not None and left_wrist[1] < nose_landmark[1]:
                        hand_on_forehead = True

            if right_wrist is not None:
                dist = np.linalg.norm(right_wrist - face_center)
                if dist < VERY_CLOSE_THRESHOLD:
                    hand_is_near_face = True
                    if nose_landmark is not None and right_wrist[1] < nose_landmark[1]:
                        hand_on_forehead = True

            # Temporal smoothing
            if hand_is_near_face:
                state.hand_near_face_count += 1
            else:
                state.hand_near_face_count = max(0, state.hand_near_face_count - 1)

            if state.hand_near_face_count >= HAND_NEAR_FACE_MIN_FRAMES:
                if hand_on_forehead:
                    info['hands_position'] = 'on_forehead'
                else:
                    info['hands_position'] = 'on_face'

        # Mãos próximas ao tronco
        if left_wrist is not None and right_wrist is not None:
            avg_wrist = (left_wrist + right_wrist) / 2

            if left_hip is not None:
                hip_y = left_hip[1]
                if neck_y < avg_wrist[1] < hip_y:
                    if np.linalg.norm(left_wrist - right_wrist) < shoulder_width:
                        info['hands_position'] = 'close_to_torso'

        # Acenando - mão acima do OMBRO + movimento + olhando pra câmera
        wave_detected = False
        if state and len(state.left_wrist_x_history) >= 5:
            looking_at_camera = (face_bbox is not None)

            if looking_at_camera and left_shoulder is not None and right_shoulder is not None:
                shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2.0

                left_above_shoulder = left_wrist is not None and left_wrist[1] < shoulder_y
                right_above_shoulder = right_wrist is not None and right_wrist[1] < shoulder_y

                # Apenas UMA mão acima do ombro (XOR)
                if left_above_shoulder ^ right_above_shoulder:
                    if left_above_shoulder and variance_left > 40:
                        wave_detected = True
                    elif right_above_shoulder and variance_right > 40:
                        wave_detected = True

        info['wave_motion'] = wave_detected

        # Compatibilidade com regras antigas
        if neck_y > 0 and (left_wrist is not None or right_wrist is not None):
            left_up = left_wrist is not None and left_wrist[1] < neck_y
            right_up = right_wrist is not None and right_wrist[1] < neck_y
            if left_up ^ right_up:
                info['one_hand_above_neck'] = True

        return info

    def _check_pose_conditions(self, pose_info: Dict, conditions: Dict) -> bool:
        """
        Verifica se as condições de pose são satisfeitas

        Args:
            pose_info: Informações de pose analisadas
            conditions: Condições requeridas

        Returns:
            True se todas as condições são satisfeitas
        """
        for key, value in conditions.items():
            if key not in pose_info:
                continue

            if isinstance(value, list):
                if pose_info[key] not in value:
                    return False
            else:
                if pose_info[key] != value:
                    return False

        return True
