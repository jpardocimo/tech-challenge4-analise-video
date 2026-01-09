"""
Módulo para análise de pose corporal usando keypoints do YOLO Pose

Responsabilidades:
- Extração de keypoints individuais
- Análise de postura corporal (em pé, sentado, deitado)
- Extração de features geométricas (distâncias, ângulos)
- Cálculo de scores de movimento (geométrico e espacial)
"""

from typing import Dict, Optional, Tuple
import numpy as np
from config.settings import KEYPOINT_MAP
from utils.geometry import calculate_distance, calculate_angle


# ============================================================================
# CONSTANTES - Thresholds e Ratios
# ============================================================================

# Thresholds de ângulos articulares
KNEE_ANGLE_SITTING_THRESHOLD = 140
KNEE_ANGLE_STANDING_THRESHOLD = 165
HIP_ANGLE_STANDING_THRESHOLD = 165
BODY_ANGLE_LYING_THRESHOLD = 60
BODY_ANGLE_VERTICAL_THRESHOLD = 15

# Ratios para detecção de posição de braços
ARM_RAISED_OFFSET_RATIO = 0.2
ARM_FORWARD_OFFSET_RATIO = 0.5

# Tolerância para alinhamento vertical de pernas
LEG_ALIGNMENT_TOLERANCE_RATIO = 0.4

# Threshold para detecção de postura em pé baseada em torso
TORSO_STANDING_RATIO_THRESHOLD = 2.0

# Detecção de movimento de pernas
KNEE_ANGLE_DIFFERENCE_WALKING = 30
KNEE_ANGLE_STRAIGHT_THRESHOLD = 160

# Confiança mínima
MIN_KEYPOINTS_FOR_CONFIDENCE = 13
MIN_VALID_KEYPOINTS_MOTION = 5


# ============================================================================
# FUNÇÕES AUXILIARES PRIVADAS
# ============================================================================

def _extract_body_keypoints(keypoints, conf_min: float) -> Dict[str, Optional[np.ndarray]]:
    """
    Extrai todos os keypoints do corpo de uma vez
    
    Args:
        keypoints: Array de keypoints do YOLO (17x3)
        conf_min: Confiança mínima
        
    Returns:
        Dicionário com todos os keypoints extraídos
    """
    return {
        'nose': get_keypoint(keypoints, 'nose', conf_min),
        'left_shoulder': get_keypoint(keypoints, 'left_shoulder', conf_min),
        'right_shoulder': get_keypoint(keypoints, 'right_shoulder', conf_min),
        'left_elbow': get_keypoint(keypoints, 'left_elbow', conf_min),
        'right_elbow': get_keypoint(keypoints, 'right_elbow', conf_min),
        'left_wrist': get_keypoint(keypoints, 'left_wrist', conf_min),
        'right_wrist': get_keypoint(keypoints, 'right_wrist', conf_min),
        'left_hip': get_keypoint(keypoints, 'left_hip', conf_min),
        'right_hip': get_keypoint(keypoints, 'right_hip', conf_min),
        'left_knee': get_keypoint(keypoints, 'left_knee', conf_min),
        'right_knee': get_keypoint(keypoints, 'right_knee', conf_min),
        'left_ankle': get_keypoint(keypoints, 'left_ankle', conf_min),
        'right_ankle': get_keypoint(keypoints, 'right_ankle', conf_min),
    }


def _calculate_reference_points(kps: Dict) -> Dict:
    """
    Calcula pontos de referência (centros e larguras)
    
    Args:
        kps: Dicionário de keypoints
        
    Returns:
        Dicionário com shoulder_center, hip_center, shoulder_width, body_angle
    """
    ref = {}
    
    # Calcula centro e largura dos ombros
    if kps['left_shoulder'] is not None and kps['right_shoulder'] is not None:
        ref['shoulder_center'] = (kps['left_shoulder'] + kps['right_shoulder']) / 2.0
        ref['shoulder_width'] = calculate_distance(kps['left_shoulder'], kps['right_shoulder'])
    else:
        ref['shoulder_center'] = None
        ref['shoulder_width'] = None
    
    # Calcula centro dos quadris
    if kps['left_hip'] is not None and kps['right_hip'] is not None:
        ref['hip_center'] = (kps['left_hip'] + kps['right_hip']) / 2.0
    else:
        ref['hip_center'] = None
    
    # Calcula ângulo do corpo (inclinação)
    if ref['shoulder_center'] is not None and ref['hip_center'] is not None:
        dy = ref['hip_center'][1] - ref['shoulder_center'][1]
        dx = ref['hip_center'][0] - ref['shoulder_center'][0]
        ref['body_angle'] = abs(np.degrees(np.arctan2(dy, dx)) - 90)  # 0° = vertical
    else:
        ref['body_angle'] = None
    
    return ref


def _calculate_confidence(kps: Dict) -> float:
    """
    Calcula confiança baseada em pontos detectados
    
    Args:
        kps: Dicionário de keypoints
        
    Returns:
        Confiança normalizada (0-1)
    """
    detected_points = sum(1 for v in kps.values() if v is not None)
    return min(1.0, detected_points / MIN_KEYPOINTS_FOR_CONFIDENCE)


def _calculate_joint_angles(kps: Dict) -> Dict[str, float]:
    """
    Calcula todos os ângulos articulares
    
    Args:
        kps: Dicionário de keypoints
        
    Returns:
        Dicionário com ângulos calculados
    """
    angles = {}
    
    # Cotovelos (shoulder-elbow-wrist)
    if kps['left_shoulder'] is not None and kps['left_elbow'] is not None and kps['left_wrist'] is not None:
        angle = calculate_angle(kps['left_shoulder'], kps['left_elbow'], kps['left_wrist'])
        if angle is not None:
            angles['left_elbow'] = angle
    
    if kps['right_shoulder'] is not None and kps['right_elbow'] is not None and kps['right_wrist'] is not None:
        angle = calculate_angle(kps['right_shoulder'], kps['right_elbow'], kps['right_wrist'])
        if angle is not None:
            angles['right_elbow'] = angle
    
    # Joelhos (hip-knee-ankle)
    if kps['left_hip'] is not None and kps['left_knee'] is not None and kps['left_ankle'] is not None:
        angle = calculate_angle(kps['left_hip'], kps['left_knee'], kps['left_ankle'])
        if angle is not None:
            angles['left_knee'] = angle
    
    if kps['right_hip'] is not None and kps['right_knee'] is not None and kps['right_ankle'] is not None:
        angle = calculate_angle(kps['right_hip'], kps['right_knee'], kps['right_ankle'])
        if angle is not None:
            angles['right_knee'] = angle
    
    # Quadris (shoulder-hip-knee)
    if kps['left_shoulder'] is not None and kps['left_hip'] is not None and kps['left_knee'] is not None:
        angle = calculate_angle(kps['left_shoulder'], kps['left_hip'], kps['left_knee'])
        if angle is not None:
            angles['left_hip'] = angle
    
    if kps['right_shoulder'] is not None and kps['right_hip'] is not None and kps['right_knee'] is not None:
        angle = calculate_angle(kps['right_shoulder'], kps['right_hip'], kps['right_knee'])
        if angle is not None:
            angles['right_hip'] = angle
    
    # Ombros (elbow-shoulder-hip)
    if kps['left_elbow'] is not None and kps['left_shoulder'] is not None and kps['left_hip'] is not None:
        angle = calculate_angle(kps['left_elbow'], kps['left_shoulder'], kps['left_hip'])
        if angle is not None:
            angles['left_shoulder'] = angle
    
    if kps['right_elbow'] is not None and kps['right_shoulder'] is not None and kps['right_hip'] is not None:
        angle = calculate_angle(kps['right_elbow'], kps['right_shoulder'], kps['right_hip'])
        if angle is not None:
            angles['right_shoulder'] = angle
    
    return angles


def _check_leg_vertical_alignment(hip, knee, ankle, shoulder_width: float) -> bool:
    """
    Verifica se uma perna está verticalmente alinhada (para detecção de standing)
    
    Args:
        hip: Coordenadas do quadril
        knee: Coordenadas do joelho
        ankle: Coordenadas do tornozelo
        shoulder_width: Largura dos ombros (para normalização)
        
    Returns:
        True se a perna está alinhada verticalmente
    """
    if hip is None or knee is None or ankle is None:
        return False
    
    ankle_below_knee = ankle[1] > knee[1]
    max_horizontal_offset = shoulder_width * LEG_ALIGNMENT_TOLERANCE_RATIO
    hip_knee_aligned = abs(hip[0] - knee[0]) < max_horizontal_offset
    knee_ankle_aligned = abs(knee[0] - ankle[0]) < max_horizontal_offset
    
    return ankle_below_knee and hip_knee_aligned and knee_ankle_aligned


def _is_lying_down(ref: Dict) -> bool:
    """
    Detecta se pessoa está deitada baseado no ângulo do corpo
    
    Args:
        ref: Dicionário de pontos de referência
        
    Returns:
        True se pessoa está deitada
    """
    body_angle = ref.get('body_angle')
    return body_angle is not None and body_angle > BODY_ANGLE_LYING_THRESHOLD


def _is_standing_strict(kps: Dict, angles: Dict, ref: Dict) -> bool:
    """
    Detecta se pessoa está em pé usando regra rigorosa
    
    Regras:
    - Joelhos e quadris retos (ângulos > 165°)
    - Pelo menos um tornozelo detectado
    - Alinhamento vertical de pelo menos uma perna
    
    Args:
        kps: Dicionário de keypoints
        angles: Dicionário de ângulos
        ref: Dicionário de pontos de referência
        
    Returns:
        True se pessoa está em pé
    """
    knee_angles = [v for k, v in angles.items() if 'knee' in k]
    hip_angles = [v for k, v in angles.items() if 'hip' in k]
    
    if not knee_angles:
        return False
    
    avg_knee = np.mean(knee_angles)
    avg_hip = np.mean(hip_angles) if hip_angles else 180
    
    # Verifica se joelhos e quadris estão retos
    if avg_knee <= KNEE_ANGLE_STANDING_THRESHOLD or avg_hip <= HIP_ANGLE_STANDING_THRESHOLD:
        return False
    
    # Verifica se tem pelo menos um tornozelo
    has_ankle = kps['left_ankle'] is not None or kps['right_ankle'] is not None
    if not has_ankle:
        return False
    
    shoulder_width = ref.get('shoulder_width', 0)
    if shoulder_width < 1e-3:
        return False
    
    # Verifica alinhamento vertical de pelo menos uma perna
    left_aligned = _check_leg_vertical_alignment(
        kps['left_hip'], kps['left_knee'], kps['left_ankle'], shoulder_width
    )
    right_aligned = _check_leg_vertical_alignment(
        kps['right_hip'], kps['right_knee'], kps['right_ankle'], shoulder_width
    )
    
    return left_aligned or right_aligned


def _is_standing_fallback(ref: Dict) -> bool:
    """
    Detecta standing usando fallback (quando não há pernas detectadas)
    
    Args:
        ref: Dicionário de pontos de referência
        
    Returns:
        True se provavelmente está em pé
    """
    shoulder_center = ref.get('shoulder_center')
    hip_center = ref.get('hip_center')
    shoulder_width = ref.get('shoulder_width', 0)
    body_angle = ref.get('body_angle')
    
    if shoulder_center is None or hip_center is None or shoulder_width < 1e-3:
        return False
    
    torso_height = abs(hip_center[1] - shoulder_center[1])
    torso_ratio = torso_height / shoulder_width
    
    return (torso_ratio > TORSO_STANDING_RATIO_THRESHOLD and 
            body_angle is not None and 
            body_angle < BODY_ANGLE_VERTICAL_THRESHOLD)


def _classify_posture(kps: Dict, angles: Dict, ref: Dict) -> str:
    """
    Classifica postura corporal
    
    Args:
        kps: Dicionário de keypoints
        angles: Dicionário de ângulos
        ref: Dicionário de pontos de referência
        
    Returns:
        'standing', 'sitting', 'lying_down', ou 'unknown'
    """
    # Early return: deitado
    if _is_lying_down(ref):
        return 'lying_down'
    
    # Detecta baseado em ângulos de joelhos
    if 'left_knee' in angles or 'right_knee' in angles:
        knee_angles = [v for k, v in angles.items() if 'knee' in k]
        avg_knee = np.mean(knee_angles)
        
        # Sentado: joelhos dobrados
        if avg_knee < KNEE_ANGLE_SITTING_THRESHOLD:
            return 'sitting'
        
        # Em pé: regra rigorosa
        if _is_standing_strict(kps, angles, ref):
            return 'standing'
        
        # Se não passou na regra rigorosa, considera sentado
        return 'sitting'
    
    # Fallback: sem pernas detectadas
    if _is_standing_fallback(ref):
        return 'standing'
    
    return 'sitting'


def _classify_arms(kps: Dict, ref: Dict) -> str:
    """
    Classifica posição dos braços
    
    Args:
        kps: Dicionário de keypoints
        ref: Dicionário de pontos de referência
        
    Returns:
        'raised', 'one_raised', 'forward', 'down', ou 'unknown'
    """
    shoulder_center = ref.get('shoulder_center')
    shoulder_width = ref.get('shoulder_width', 0)
    
    if shoulder_center is None or shoulder_width < 1e-3:
        return 'unknown'
    
    shoulder_y = shoulder_center[1]
    left_arm_raised = False
    right_arm_raised = False
    left_arm_forward = False
    right_arm_forward = False
    
    # Detecta braços levantados
    if kps['left_wrist'] is not None:
        if kps['left_wrist'][1] < shoulder_y - shoulder_width * ARM_RAISED_OFFSET_RATIO:
            left_arm_raised = True
        elif kps['left_wrist'][1] < shoulder_y + shoulder_width * ARM_FORWARD_OFFSET_RATIO:
            left_arm_forward = True
    
    if kps['right_wrist'] is not None:
        if kps['right_wrist'][1] < shoulder_y - shoulder_width * ARM_RAISED_OFFSET_RATIO:
            right_arm_raised = True
        elif kps['right_wrist'][1] < shoulder_y + shoulder_width * ARM_FORWARD_OFFSET_RATIO:
            right_arm_forward = True
    
    # Classifica baseado em combinações
    if left_arm_raised and right_arm_raised:
        return 'raised'
    if left_arm_raised or right_arm_raised:
        return 'one_raised'
    if left_arm_forward or right_arm_forward:
        return 'forward'
    if kps['left_wrist'] is not None or kps['right_wrist'] is not None:
        return 'down'
    
    return 'unknown'


def _classify_legs(angles: Dict) -> str:
    """
    Classifica estado das pernas
    
    Args:
        angles: Dicionário de ângulos
        
    Returns:
        'standing', 'walking', 'bent', ou 'unknown'
    """
    if 'left_knee' not in angles or 'right_knee' not in angles:
        return 'unknown'
    
    left_knee_angle = angles['left_knee']
    right_knee_angle = angles['right_knee']
    
    # Pernas retas
    if left_knee_angle > KNEE_ANGLE_STRAIGHT_THRESHOLD and right_knee_angle > KNEE_ANGLE_STRAIGHT_THRESHOLD:
        return 'standing'
    
    # Andando: diferença significativa entre joelhos
    if abs(left_knee_angle - right_knee_angle) > KNEE_ANGLE_DIFFERENCE_WALKING:
        return 'walking'
    
    return 'bent'


def _build_details(ref: Dict, kps: Dict) -> Dict:
    """
    Constrói dicionário de detalhes para debug
    
    Args:
        ref: Dicionário de pontos de referência
        kps: Dicionário de keypoints
        
    Returns:
        Dicionário com informações detalhadas
    """
    details = {}
    
    if ref.get('body_angle') is not None:
        details['body_angle'] = ref['body_angle']
    
    if ref.get('shoulder_width') is not None:
        details['shoulder_width'] = ref['shoulder_width']
    
    # Calcula torso_ratio se possível
    shoulder_center = ref.get('shoulder_center')
    hip_center = ref.get('hip_center')
    shoulder_width = ref.get('shoulder_width', 0)
    
    if shoulder_center is not None and hip_center is not None and shoulder_width > 0:
        torso_height = abs(hip_center[1] - shoulder_center[1])
        details['torso_ratio'] = torso_height / shoulder_width
    
    detected_points = sum(1 for v in kps.values() if v is not None)
    details['detected_points'] = detected_points
    
    return details


# ============================================================================
# FUNÇÕES PÚBLICAS
# ============================================================================

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
    # Inicializa resultado
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

    # 1. Extrai todos os keypoints
    kps = _extract_body_keypoints(keypoints, conf_min)

    # 2. Calcula pontos de referência
    ref = _calculate_reference_points(kps)

    # Validação mínima: precisa de ombros
    if ref['shoulder_center'] is None or ref.get('shoulder_width', 0) < 1e-3:
        return result

    # 3. Calcula confiança
    result['confidence'] = _calculate_confidence(kps)

    # 4. Calcula ângulos articulares
    result['angles'] = _calculate_joint_angles(kps)

    # 5. Classifica postura
    result['posture'] = _classify_posture(kps, result['angles'], ref)

    # 6. Classifica braços
    result['arms'] = _classify_arms(kps, ref)

    # 7. Classifica pernas
    result['legs'] = _classify_legs(result['angles'])

    # 8. Adiciona detalhes
    result['details'] = _build_details(ref, kps)

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
    Calcula score de movimento geométrico baseado em variação de features de pose

    Args:
        previous_features: Features do frame anterior
        current_features: Features do frame atual
        delta_time: Intervalo de tempo entre frames (segundos)

    Returns:
        Score de movimento (velocidade agregada de mudança geométrica) ou None
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


def calculate_keypoint_motion_variance(
    previous_keypoints: np.ndarray,
    current_keypoints: np.ndarray,
    delta_time: float,
    conf_min: float = 0.5
) -> Optional[float]:
    """
    Calcula variância de movimento dos keypoints (detecção espacial).
    Robusto contra movimento de câmera (pan/zoom).

    Se todos os keypoints se movem uniformemente = câmera movendo (variância baixa)
    Se keypoints têm movimentos variados = pessoa se movendo bruscamente (variância alta)

    Args:
        previous_keypoints: Array (17, 3) do frame anterior
        current_keypoints: Array (17, 3) do frame atual
        delta_time: Tempo entre frames (segundos)
        conf_min: Confiança mínima para considerar keypoint

    Returns:
        Variância normalizada do movimento (quanto maior, mais heterogêneo) ou None
    """
    if previous_keypoints is None or current_keypoints is None or delta_time <= 0:
        return None

    # Extrai keypoints válidos
    prev_valid = []
    curr_valid = []

    for i in range(min(17, previous_keypoints.shape[0])):
        if (previous_keypoints[i, 2] >= conf_min and
            current_keypoints[i, 2] >= conf_min):
            prev_valid.append(previous_keypoints[i, :2])
            curr_valid.append(current_keypoints[i, :2])

    if len(prev_valid) < MIN_VALID_KEYPOINTS_MOTION:
        return None

    prev_valid = np.array(prev_valid)
    curr_valid = np.array(curr_valid)

    # Calcula deslocamento de cada keypoint
    displacements = curr_valid - prev_valid
    velocities = np.linalg.norm(displacements, axis=1) / delta_time

    # Calcula média e variância
    mean_velocity = np.mean(velocities)
    velocity_variance = np.var(velocities)

    # Normaliza pela média para ter escala consistente
    if mean_velocity > 1e-3:
        normalized_variance = velocity_variance / (mean_velocity + 1e-6)
    else:
        normalized_variance = 0.0

    return float(normalized_variance)
