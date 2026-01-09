"""
Módulo para análise de pose corporal usando keypoints do YOLO Pose
"""

from typing import Dict, Optional
import numpy as np
from config.settings import KEYPOINT_MAP
from utils.geometry import calculate_distance, calculate_angle


def get_keypoint(keypoints, part_name: str, conf_min: float = 0.5) -> Optional[np.ndarray]:
    """
    Retorna coordenadas [x,y] de um keypoint pelo nome semântico

    Args:
        keypoints: Array de keypoints do YOLO (Nx3: x, y, confidence)
        part_name: Nome da parte do corpo (ex: 'left_knee', 'nose', 'right_wrist')
        conf_min: Confiança mínima (0.0-1.0)

    Returns:
        np.array([x, y], dtype=float32) ou None se não detectado/baixa confiança

    Exemplos:
        >>> nose = get_keypoint(keypoints, 'nose')
        >>> left_knee = get_keypoint(keypoints, 'left_knee', conf_min=0.6)
    """
    if keypoints is None:
        return None

    idx = KEYPOINT_MAP.get(part_name)
    if idx is None:
        return None

    if idx >= len(keypoints):
        return None

    if float(keypoints[idx][2]) < conf_min:
        return None

    return np.array([float(keypoints[idx][0]), float(keypoints[idx][1])], dtype=np.float32)


def analyze_body_pose(keypoints, conf_min: float = 0.5) -> Dict:
    """
    Analisa a pose do corpo e retorna classificação baseada em boas práticas da indústria

    Detecta poses usando:
    - Ângulos articulares (cotovelos, joelhos, quadris, ombros)
    - Relações posicionais (eixo Y)
    - Distâncias normalizadas pela largura dos ombros

    Args:
        keypoints: Array de keypoints do YOLO (17x3)
        conf_min: Confiança mínima para considerar keypoint válido

    Returns:
        dict com estrutura:
        {
            'posture': str - 'standing', 'sitting', 'squatting', 'lying_down', 'unknown',
            'arms': str - 'raised', 'forward', 'down', 'one_raised', 'unknown',
            'legs': str - 'standing', 'walking', 'bent', 'unknown',
            'angles': dict - ângulos calculados (graus),
            'confidence': float - confiança geral da detecção (0-1),
            'details': dict - informações detalhadas para debug
        }
    """
    result = {
        'posture': 'unknown',
        'arms': 'unknown',
        'legs': 'unknown',
        'angles': {},
        'confidence': 0.0,
        'details': {}
    }

    if keypoints is None:
        return result

    # Extrai keypoints principais
    nose = get_keypoint(keypoints, 'nose', conf_min)
    left_shoulder = get_keypoint(keypoints, 'left_shoulder', conf_min)
    right_shoulder = get_keypoint(keypoints, 'right_shoulder', conf_min)
    left_elbow = get_keypoint(keypoints, 'left_elbow', conf_min)
    right_elbow = get_keypoint(keypoints, 'right_elbow', conf_min)
    left_wrist = get_keypoint(keypoints, 'left_wrist', conf_min)
    right_wrist = get_keypoint(keypoints, 'right_wrist', conf_min)
    left_hip = get_keypoint(keypoints, 'left_hip', conf_min)
    right_hip = get_keypoint(keypoints, 'right_hip', conf_min)
    left_knee = get_keypoint(keypoints, 'left_knee', conf_min)
    right_knee = get_keypoint(keypoints, 'right_knee', conf_min)
    left_ankle = get_keypoint(keypoints, 'left_ankle', conf_min)
    right_ankle = get_keypoint(keypoints, 'right_ankle', conf_min)

    # Calcula pontos de referência
    shoulder_center = None
    hip_center = None
    shoulder_width = None

    if left_shoulder is not None and right_shoulder is not None:
        shoulder_center = (left_shoulder + right_shoulder) / 2.0
        shoulder_width = calculate_distance(left_shoulder, right_shoulder)

    if left_hip is not None and right_hip is not None:
        hip_center = (left_hip + right_hip) / 2.0

    # Validação mínima: precisa de ombros
    if shoulder_center is None or shoulder_width < 1e-3:
        return result

    # Calcula confiança baseada em pontos detectados
    detected_points = sum([
        nose is not None, left_shoulder is not None, right_shoulder is not None,
        left_elbow is not None, right_elbow is not None, left_wrist is not None,
        right_wrist is not None, left_hip is not None, right_hip is not None,
        left_knee is not None, right_knee is not None, left_ankle is not None,
        right_ankle is not None
    ])
    result['confidence'] = min(1.0, detected_points / 13.0)

    # =========================================================================
    # ANÁLISE DE ÂNGULOS ARTICULARES
    # =========================================================================

    # Cotovelos (shoulder-elbow-wrist)
    if left_shoulder is not None and left_elbow is not None and left_wrist is not None:
        angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        if angle is not None:
            result['angles']['left_elbow'] = angle

    if right_shoulder is not None and right_elbow is not None and right_wrist is not None:
        angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        if angle is not None:
            result['angles']['right_elbow'] = angle

    # Joelhos (hip-knee-ankle)
    if left_hip is not None and left_knee is not None and left_ankle is not None:
        angle = calculate_angle(left_hip, left_knee, left_ankle)
        if angle is not None:
            result['angles']['left_knee'] = angle

    if right_hip is not None and right_knee is not None and right_ankle is not None:
        angle = calculate_angle(right_hip, right_knee, right_ankle)
        if angle is not None:
            result['angles']['right_knee'] = angle

    # Quadris (shoulder-hip-knee)
    if left_shoulder is not None and left_hip is not None and left_knee is not None:
        angle = calculate_angle(left_shoulder, left_hip, left_knee)
        if angle is not None:
            result['angles']['left_hip'] = angle

    if right_shoulder is not None and right_hip is not None and right_knee is not None:
        angle = calculate_angle(right_shoulder, right_hip, right_knee)
        if angle is not None:
            result['angles']['right_hip'] = angle

    # Ombros (elbow-shoulder-hip)
    if left_elbow is not None and left_shoulder is not None and left_hip is not None:
        angle = calculate_angle(left_elbow, left_shoulder, left_hip)
        if angle is not None:
            result['angles']['left_shoulder'] = angle

    if right_elbow is not None and right_shoulder is not None and right_hip is not None:
        angle = calculate_angle(right_elbow, right_shoulder, right_hip)
        if angle is not None:
            result['angles']['right_shoulder'] = angle

    # =========================================================================
    # CLASSIFICAÇÃO DE POSTURA (POSTURE)
    # =========================================================================

    # Calcula ângulo do corpo (inclinação)
    body_angle = None
    if shoulder_center is not None and hip_center is not None:
        dy = hip_center[1] - shoulder_center[1]
        dx = hip_center[0] - shoulder_center[0]
        body_angle = abs(np.degrees(np.arctan2(dy, dx)) - 90)  # 0° = vertical
        result['details']['body_angle'] = body_angle

    # Detecta LYING DOWN (deitado)
    if body_angle is not None and body_angle > 60:  # > 60° de inclinação
        result['posture'] = 'lying_down'

    # Detecta SITTING, SQUATTING, STANDING baseado em joelhos e quadris
    elif 'left_knee' in result['angles'] or 'right_knee' in result['angles']:
        knee_angles = [v for k, v in result['angles'].items() if 'knee' in k]
        hip_angles = [v for k, v in result['angles'].items() if 'hip' in k]

        avg_knee = np.mean(knee_angles) if knee_angles else 180
        avg_hip = np.mean(hip_angles) if hip_angles else 180

        # Calcula torso_ratio para validação adicional
        torso_height = abs(hip_center[1] - shoulder_center[1]) if hip_center is not None else 0
        torso_ratio = torso_height / shoulder_width if shoulder_width > 0 else 0
        result['details']['torso_ratio'] = torso_ratio

        # SITTING: joelhos dobrados
        if avg_knee < 140:
            result['posture'] = 'sitting'

        # STANDING: REGRA RIGOROSA
        elif avg_knee > 165 and avg_hip > 165:
            has_ankle = left_ankle is not None or right_ankle is not None

            if has_ankle:
                vertical_alignment_ok = False

                # Verifica perna esquerda
                if left_hip is not None and left_knee is not None and left_ankle is not None:
                    ankle_below_knee = left_ankle[1] > left_knee[1]
                    max_horizontal_offset = shoulder_width * 0.4
                    hip_knee_aligned = abs(left_hip[0] - left_knee[0]) < max_horizontal_offset
                    knee_ankle_aligned = abs(left_knee[0] - left_ankle[0]) < max_horizontal_offset

                    if ankle_below_knee and hip_knee_aligned and knee_ankle_aligned:
                        vertical_alignment_ok = True

                # Verifica perna direita
                if not vertical_alignment_ok and right_hip is not None and right_knee is not None and right_ankle is not None:
                    ankle_below_knee = right_ankle[1] > right_knee[1]
                    max_horizontal_offset = shoulder_width * 0.4
                    hip_knee_aligned = abs(right_hip[0] - right_knee[0]) < max_horizontal_offset
                    knee_ankle_aligned = abs(right_knee[0] - right_ankle[0]) < max_horizontal_offset

                    if ankle_below_knee and hip_knee_aligned and knee_ankle_aligned:
                        vertical_alignment_ok = True

                if vertical_alignment_ok:
                    result['posture'] = 'standing'
                else:
                    result['posture'] = 'sitting'
            else:
                result['posture'] = 'sitting'
        else:
            result['posture'] = 'sitting'

    # FALLBACK: Sem pernas detectadas
    elif shoulder_center is not None and hip_center is not None:
        torso_height = abs(hip_center[1] - shoulder_center[1])
        torso_ratio = torso_height / shoulder_width if shoulder_width > 0 else 0
        result['details']['torso_ratio'] = torso_ratio

        if torso_ratio > 2.0 and body_angle is not None and body_angle < 15:
            result['posture'] = 'standing'
        else:
            result['posture'] = 'sitting'

    # =========================================================================
    # CLASSIFICAÇÃO DE BRAÇOS (ARMS)
    # =========================================================================

    left_arm_raised = False
    right_arm_raised = False
    left_arm_forward = False
    right_arm_forward = False

    if shoulder_center is not None:
        shoulder_y = shoulder_center[1]

        # Detecta braços levantados
        if left_wrist is not None:
            if left_wrist[1] < shoulder_y - shoulder_width * 0.2:
                left_arm_raised = True
            elif left_wrist[1] < shoulder_y + shoulder_width * 0.5:
                left_arm_forward = True

        if right_wrist is not None:
            if right_wrist[1] < shoulder_y - shoulder_width * 0.2:
                right_arm_raised = True
            elif right_wrist[1] < shoulder_y + shoulder_width * 0.5:
                right_arm_forward = True

    # Classifica baseado em combinações
    if left_arm_raised and right_arm_raised:
        result['arms'] = 'raised'
    elif left_arm_raised or right_arm_raised:
        result['arms'] = 'one_raised'
    elif left_arm_forward or right_arm_forward:
        result['arms'] = 'forward'
    elif left_wrist is not None or right_wrist is not None:
        result['arms'] = 'down'

    # =========================================================================
    # CLASSIFICAÇÃO DE PERNAS (LEGS)
    # =========================================================================

    if 'left_knee' in result['angles'] and 'right_knee' in result['angles']:
        left_knee_angle = result['angles']['left_knee']
        right_knee_angle = result['angles']['right_knee']

        if left_knee_angle > 160 and right_knee_angle > 160:
            result['legs'] = 'standing'
        elif abs(left_knee_angle - right_knee_angle) > 30:
            result['legs'] = 'walking'
        else:
            result['legs'] = 'bent'

    result['details']['shoulder_width'] = shoulder_width
    result['details']['detected_points'] = detected_points

    return result


def extract_pose_features(keypoints, conf_min: float = 0.5) -> Optional[Dict]:
    """
    Extrai features normalizadas da pose (distâncias e ângulos)
    Robustas contra movimento de câmera

    Args:
        keypoints: Array de keypoints do YOLO (17x3)
        conf_min: Confiança mínima para keypoints

    Returns:
        Dicionário com features ou None se não há dados suficientes
        {
            'd_lwr_lsh': distância normalizada,
            'ang_l_elbow': ângulo em graus,
            ...
            '_shoulder_w': largura dos ombros (referência)
        }
    """
    nose = get_keypoint(keypoints, 'nose', conf_min)
    left_shoulder = get_keypoint(keypoints, 'left_shoulder', conf_min)
    right_shoulder = get_keypoint(keypoints, 'right_shoulder', conf_min)
    left_elbow = get_keypoint(keypoints, 'left_elbow', conf_min)
    right_elbow = get_keypoint(keypoints, 'right_elbow', conf_min)
    left_wrist = get_keypoint(keypoints, 'left_wrist', conf_min)
    right_wrist = get_keypoint(keypoints, 'right_wrist', conf_min)
    left_hip = get_keypoint(keypoints, 'left_hip', conf_min)
    right_hip = get_keypoint(keypoints, 'right_hip', conf_min)

    if left_shoulder is None or right_shoulder is None:
        return None

    shoulder_width = calculate_distance(left_shoulder, right_shoulder)
    if shoulder_width < 1e-3:
        return None

    features = {}

    # Distâncias normalizadas
    if left_wrist is not None:
        features["d_lwr_lsh"] = calculate_distance(left_wrist, left_shoulder) / shoulder_width
        if nose is not None:
            features["d_lwr_nose"] = calculate_distance(left_wrist, nose) / shoulder_width

    if right_wrist is not None:
        features["d_rwr_rsh"] = calculate_distance(right_wrist, right_shoulder) / shoulder_width
        if nose is not None:
            features["d_rwr_nose"] = calculate_distance(right_wrist, nose) / shoulder_width

    if left_wrist is not None and right_wrist is not None:
        features["d_wrists"] = calculate_distance(left_wrist, right_wrist) / shoulder_width

    # Ângulos dos cotovelos
    if left_shoulder is not None and left_elbow is not None and left_wrist is not None:
        angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        if angle is not None:
            features["ang_l_elbow"] = angle

    if right_shoulder is not None and right_elbow is not None and right_wrist is not None:
        angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        if angle is not None:
            features["ang_r_elbow"] = angle

    # Ângulos dos ombros
    if left_elbow is not None and left_shoulder is not None and left_hip is not None:
        angle = calculate_angle(left_elbow, left_shoulder, left_hip)
        if angle is not None:
            features["ang_l_shoulder"] = angle

    if right_elbow is not None and right_shoulder is not None and right_hip is not None:
        angle = calculate_angle(right_elbow, right_shoulder, right_hip)
        if angle is not None:
            features["ang_r_shoulder"] = angle

    # Validação: precisa de pelo menos 1 ângulo ou 2 distâncias
    has_elbow = ("ang_l_elbow" in features) or ("ang_r_elbow" in features)
    dist_count = len([k for k in features.keys() if k.startswith("d_")])

    if not has_elbow and dist_count < 2:
        return None

    features["_shoulder_w"] = shoulder_width
    return features


def calculate_pose_motion_score(previous_features: Dict, current_features: Dict, delta_time: float) -> Optional[float]:
    """
    Calcula score de movimento baseado em variação de features

    Args:
        previous_features: Features do frame anterior
        current_features: Features do frame atual
        delta_time: Intervalo de tempo entre frames (segundos)

    Returns:
        Score de movimento (velocidade agregada) ou None
    """
    if previous_features is None or current_features is None or delta_time <= 0:
        return None

    dist_keys = [k for k in current_features.keys() if k.startswith("d_")]
    ang_keys = [k for k in current_features.keys() if k.startswith("ang_")]

    dist_rates = []
    for k in dist_keys:
        if k in previous_features:
            dist_rates.append(abs(current_features[k] - previous_features[k]) / delta_time)

    ang_rates = []
    for k in ang_keys:
        if k in previous_features:
            ang_rates.append(abs(current_features[k] - previous_features[k]) / delta_time)

    if not dist_rates and not ang_rates:
        return None

    dist_part = float(np.mean(dist_rates)) if dist_rates else 0.0
    ang_part = float(np.mean(ang_rates)) if ang_rates else 0.0

    # Peso para balancear distâncias normalizadas e ângulos
    score = dist_part + 0.02 * ang_part
    return score
