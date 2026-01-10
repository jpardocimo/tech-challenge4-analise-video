"""
Configurações globais do sistema de análise de vídeo
"""

from datetime import datetime

# =============================================================================
# CAMINHOS DE ARQUIVOS
# =============================================================================

INPUT_VIDEO = "src/input.mp4"
ACTIONS_CONFIG_PATH = "src/config/actions_config.json"

# Pasta de saída
OUTPUT_DIR = "output"

# Gera timestamp único para os arquivos de saída
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_VIDEO = f"{OUTPUT_DIR}/video_final_{TIMESTAMP}.mp4"
OUTPUT_REPORT = f"{OUTPUT_DIR}/relatorio_final_{TIMESTAMP}.txt"
OUTPUT_LOG_CSV = f"{OUTPUT_DIR}/overlay_log_{TIMESTAMP}.csv"

# =============================================================================
# CAMINHOS DE MODELOS
# =============================================================================

YOLO_POSE_MODEL_PATH = "ai_models/yolo11s-pose.pt"
YOLO_OBJECT_MODEL_PATH = "ai_models/yolo11s.pt" 
INSIGHTFACE_MODEL_NAME = "buffalo_l"

# =============================================================================
# PERFORMANCE E PROCESSAMENTO
# =============================================================================

# Processa IA a cada N frames (para otimização de performance)
SKIP_FRAMES = 2

# Frequência de logging no CSV
LOG_EVERY_N_FRAMES = 1

# Intervalo de flush do buffer de log (frames)
LOG_FLUSH_INTERVAL = 30

# =============================================================================
# DETECÇÃO E ANÁLISE DE KEYPOINTS, OBJETOS, PESSOAS E FACES 
# =============================================================================

# Confiança mínima para detecção de keypoints
KEYPOINT_MIN_CONFIDENCE = 0.5

# Confiança mínima para detecção de pessoas
PERSON_DETECTION_CONFIDENCE = 0.5

# Confiança mínima para detecção de faces
FACE_DETECTION_CONFIDENCE = 0.5

# Confiança mínima para detecção de objetos
OBJECT_DETECTION_CONFIDENCE = 0.4

# IoU threshold para tracking de pessoas
PERSON_TRACKING_IOU = 0.3

# Threshold de similaridade para reconhecimento facial
FACE_SIMILARITY_THRESHOLD = 0.5

# =============================================================================
# ATIVIDADES E AÇÕES
# =============================================================================

# Confiança mínima de keypoints para classificação de atividades
ACTION_MIN_KEYPOINT_SCORE = 0.35

# Frames de espera antes de mudar atividade
ACTION_HOLD_FRAMES = 8

# Frames necessários para confirmar nova atividade
ACTION_CONFIRM_FRAMES = 4

# Flag para exibir postura no label de atividade
# True: Mostra "Sentado - Lendo" | False: Mostra apenas "Lendo"
SHOW_POSTURE_IN_ACTIVITY_LABEL = False

# Atividade padrão quando nenhuma atividade específica é detectada
DEFAULT_ACTIVITY_LABEL = "Atividade nao detectada"

# Labels de atividades baseadas em postura (fallback)
POSTURE_ACTIVITY_LABELS = {
    'lying_down': 'Deitado',
    'sitting': 'Sentado',
    'standing': 'Em pe'
}

# Mensagens de anomalias
ANOMALY_MESSAGES = {
    'fall_detected': 'POSSIVEL QUEDA DETECTADA',
    'abrupt_movement': 'MOVIMENTO ABRUPTO'
}

# =============================================================================
# DETECÇÃO DE ANOMALIAS
# =============================================================================

# Janela para detecção de anomalias de movimento (frames)
ANOMALY_WINDOW_SIZE = 10

# Multiplicador de desvio padrão para threshold de anomalia
ANOMALY_STD_MULTIPLIER = 3.5

# Janela para movimento de pose (frames)
POSE_MOTION_WINDOW = 60

# Multiplicador K para detecção de movimento abrupto de geometria (pose)
POSE_MOTION_K = 3

# Multiplicador K para detecção de movimento espacial (variância de keypoints)
POSE_MOTION_K_SPATIAL = 3

# Cooldown entre detecções de anomalia (frames)
ANOMALY_COOLDOWN_FRAMES = 10

# =============================================================================
# DETECÇÃO DE EMOÇÕES
# =============================================================================

# Intervalo de cache de emoções (frames)
EMOTION_CACHE_INTERVAL = 2

# Padding para crop da face (percentual)
FACE_CROP_PADDING = 0.5

# Tamanho mínimo da face para análise de emoção
MIN_FACE_SIZE = 48

# Janela de histórico de emoções (frames)
EMOTION_HISTORY_WINDOW = 30

# Threshold para detectar instabilidade emocional (mudanças)
EMOTION_INSTABILITY_THRESHOLD = 7

# =============================================================================
# SUAVIZAÇÃO E TEMPORAL
# =============================================================================

# Alpha para suavização exponencial de keypoints
KEYPOINT_SMOOTHING_ALPHA = 0.6

# Janela para histórico de posição das mãos (acenar)
HAND_POSITION_HISTORY_WINDOW = 18

# Frames mínimos para confirmar mão no rosto
HAND_NEAR_FACE_MIN_FRAMES = 2

# =============================================================================
# MAPEAMENTO DE KEYPOINTS COCO (YOLO11 Pose - 17 pontos)
# =============================================================================

KEYPOINT_MAP = {
    'nose': 0,
    'left_eye': 1, 'right_eye': 2,
    'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6,
    'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10,
    'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14,
    'left_ankle': 15, 'right_ankle': 16
}

# =============================================================================
# MAPEAMENTO DE POSTURAS
# =============================================================================

POSTURE_LABELS = {
    'standing': 'Em pe',
    'sitting': 'Sentado',
    'squatting': 'Agachado',
    'lying_down': 'Deitado',
    'unknown': ''
}

# =============================================================================
# CLASSES DE OBJETOS DE INTERESSE (COCO dataset)
# =============================================================================

INTEREST_OBJECT_CLASSES = [
    67,  # cell phone
    73,  # book
    63,  # laptop
    41,  # cup
    39,  # bottle
    60,  # dining table (mesa)
    56   # chair (cadeira)
]

# =============================================================================
# SKELETON LINKS (conexões entre keypoints para desenho)
# =============================================================================

SKELETON_LINKS = [
    (5, 7), (7, 9), (5, 9),      # Braço esquerdo
    (6, 8), (8, 10), (6, 10),    # Braço direito
    (5, 6),                       # Ombros
    (5, 11), (6, 12), (11, 12),  # Tronco
    (11, 13), (13, 15),          # Perna esquerda
    (12, 14), (14, 16)           # Perna direita
]
