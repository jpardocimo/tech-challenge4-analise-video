"""
Módulo para detecção de anomalias de movimento

Responsabilidades:
- Detectar anomalias estatísticas usando média móvel e desvio padrão
- Manter estado de movimento com detecção híbrida (geométrica + espacial)
- Não contém cálculos de pose (importa de pose_analysis)
"""

from typing import Tuple, Optional
import numpy as np
from config.settings import ANOMALY_WINDOW_SIZE, ANOMALY_STD_MULTIPLIER


class RollingAnomalyDetector:
    """
    Detecta anomalias estatísticas usando média móvel e desvio padrão

    Usa janela deslizante para calcular baseline e detectar valores
    que excedem k desvios padrão da média.
    """

    def __init__(self, window: int = ANOMALY_WINDOW_SIZE, k: float = ANOMALY_STD_MULTIPLIER):
        """
        Args:
            window: Tamanho da janela de histórico
            k: Multiplicador de desvio padrão para threshold
        """
        self.window = window
        self.k = k
        self.values = []

    def update(self, value: float) -> Tuple[bool, Optional[float], Optional[float]]:
        """
        Atualiza detector com novo valor e verifica se é anomalia

        Args:
            value: Novo valor a ser analisado

        Returns:
            Tupla (is_anomaly, mean, threshold)
            - is_anomaly: True se valor é anomalia
            - mean: Média da janela
            - threshold: Threshold calculado (mean + k*std)
        """
        self.values.append(value)

        # Remove valores antigos
        if len(self.values) > self.window:
            self.values.pop(0)

        # Precisa de pelo menos 1/3 da janela para calcular
        min_samples = max(10, self.window // 3)
        if len(self.values) < min_samples:
            return False, None, None

        array = np.array(self.values, dtype=np.float32)
        mean = float(array.mean())
        std = float(array.std() + 1e-6)  # Adiciona epsilon para evitar divisão por zero
        threshold = mean + self.k * std

        is_anomaly = value > threshold

        return is_anomaly, mean, threshold


class PoseMotionState:
    """
    Rastreia estado de movimento de pose para detecção de movimentos abruptos.
    Suporta dois tipos de detecção:
    - Geométrica (mudança de pose)
    - Espacial (movimento físico heterogêneo)

    Mantém histórico de features de pose e detecta spikes de movimento
    usando RollingAnomalyDetector.
    """

    def __init__(self, window: int = 30, k_geometric: float = 2.2, k_spatial: float = 2.5):
        """
        Args:
            window: Tamanho da janela para detecção
            k_geometric: Multiplicador de desvio padrão para pose
            k_spatial: Multiplicador de desvio padrão para movimento espacial
        """
        # Detector geométrico (mudança de pose)
        self.detector = RollingAnomalyDetector(window=window, k=k_geometric)

        # Detector espacial (movimento físico)
        self.spatial_detector = RollingAnomalyDetector(window=window, k=k_spatial)

        self.previous_features = None
        self.previous_keypoints = None
        self.last_seen_frame = -1
        self.last_anomaly_frame = -10
        self.spike_count = 0
        self.spatial_spike_count = 0


def process_pose_anomalies(
    render_data: list,
    pose_motion_states: dict,
    frame_count: int,
    delta_time: float,
    fps: float,
    pose_motion_window: int,
    pose_motion_k_geometric: float,
    pose_motion_k_spatial: float,
    anomaly_cooldown_frames: int
) -> list:
    """
    Processa detecção de anomalias de movimento para todos os itens detectados.

    Args:
        render_data: Lista de pessoas detectadas com dados
        pose_motion_states: Dicionário de estados de pose por track_id
        frame_count: Número do frame atual
        delta_time: Tempo entre frames (segundos)
        fps: Frames por segundo
        pose_motion_window: Janela para detecção
        pose_motion_k_geometric: K para detecção geométrica
        pose_motion_k_spatial: K para detecção espacial
        anomaly_cooldown_frames: Cooldown entre detecções

    Returns:
        Lista de mensagens de anomalias detectadas
    """
    from core.pose_analysis import extract_pose_features, calculate_pose_motion_score, calculate_keypoint_motion_variance

    anomalies_list = []

    for item_index, item in enumerate(render_data):
        # Obtém ID de tracking
        track_id = item.get("track_id", None)
        if track_id is None:
            face_id = item.get("face_id", None)
            track_id = int(face_id) if face_id is not None else (-1000 - item_index)

        keypoints = item.get("kps", None)
        if keypoints is None:
            continue

        # Extrai features de pose
        features = extract_pose_features(keypoints, conf_min=0.45)
        if features is None:
            continue

        # Obtém ou cria estado de pose
        pose_state = pose_motion_states.get(track_id)
        if pose_state is None:
            pose_state = PoseMotionState(
                window=pose_motion_window,
                k_geometric=pose_motion_k_geometric,
                k_spatial=pose_motion_k_spatial
            )
            pose_motion_states[track_id] = pose_state

        # ============== DETECÇÃO GEOMÉTRICA (mudança de pose) ==============
        geometric_score = calculate_pose_motion_score(
            pose_state.previous_features,
            features,
            delta_time
        )

        is_geometric_anomaly = False
        geometric_delta = 0.0

        if geometric_score is not None:
            is_anom_g, mean_g, thr_g = pose_state.detector.update(geometric_score)

            if mean_g is not None and thr_g is not None:
                geometric_delta = geometric_score - thr_g

                if is_anom_g:
                    pose_state.spike_count += 1
                else:
                    pose_state.spike_count = 0

                is_geometric_anomaly = (pose_state.spike_count >= 1) or (geometric_delta > 10.0)

        # ============== DETECÇÃO ESPACIAL (movimento físico heterogêneo) ==============
        spatial_score = calculate_keypoint_motion_variance(
            pose_state.previous_keypoints,
            keypoints,
            delta_time,
            conf_min=0.45
        )

        is_spatial_anomaly = False
        spatial_delta = 0.0

        if spatial_score is not None:
            is_anom_s, mean_s, thr_s = pose_state.spatial_detector.update(spatial_score)

            if mean_s is not None and thr_s is not None:
                spatial_delta = spatial_score - thr_s

                if is_anom_s:
                    pose_state.spatial_spike_count += 1
                else:
                    pose_state.spatial_spike_count = 0

                is_spatial_anomaly = (pose_state.spatial_spike_count >= 1) or (spatial_delta > 5.0)

        # ============== DECISÃO FINAL ==============
        is_anomaly_final = is_geometric_anomaly or is_spatial_anomaly

        # Atualiza estado
        pose_state.previous_features = features
        # Converte tensor para numpy antes de armazenar
        if hasattr(keypoints, 'cpu'):
            pose_state.previous_keypoints = keypoints.cpu().numpy()
        else:
            pose_state.previous_keypoints = keypoints.copy() if hasattr(keypoints, 'copy') else keypoints
        pose_state.last_seen_frame = frame_count

        # Debug info
        if geometric_score is not None and spatial_score is not None:
            item["pose_dbg"] = (
                f"geom={geometric_score:.2f}(Δ={geometric_delta:.1f}) "
                f"spat={spatial_score:.2f}(Δ={spatial_delta:.1f}) "
                f"spk={pose_state.spike_count}/{pose_state.spatial_spike_count}"
            )
        elif geometric_score is not None:
            item["pose_dbg"] = f"geom={geometric_score:.2f} spat=warmup"
        else:
            item["pose_dbg"] = "poseScore=-- (warmup)"

        # Dispara anomalia com cooldown
        if is_anomaly_final and (frame_count - pose_state.last_anomaly_frame) > anomaly_cooldown_frames:
            pose_state.last_anomaly_frame = frame_count
            pose_state.spike_count = 0
            pose_state.spatial_spike_count = 0

            # Identifica tipo de anomalia
            anomaly_type = []
            if is_geometric_anomaly:
                anomaly_type.append("POSE")
            if is_spatial_anomaly:
                anomaly_type.append("MOVIMENTO")

            anomaly_msg = f"MOVIMENTO ABRUPTO ({'+'.join(anomaly_type)} T{track_id})"

            if anomaly_msg not in item.get("anomalies", []):
                if "anomalies" not in item:
                    item["anomalies"] = []
                item["anomalies"].append(anomaly_msg)

            if anomaly_msg not in anomalies_list:
                anomalies_list.append(anomaly_msg)

    return anomalies_list


def cleanup_old_states(
    pose_motion_states: dict,
    action_states: dict,
    frame_count: int,
    max_frames_inactive: int = 90
):
    """
    Remove estados antigos de rastreamento.

    Remove estados que não foram atualizados há mais de max_frames_inactive frames.
    Também remove action_states órfãos (sem pose_state correspondente).

    Args:
        pose_motion_states: Dicionário {track_id: PoseMotionState}
        action_states: Dicionário {track_id: ActionState}
        frame_count: Frame atual
        max_frames_inactive: Máximo de frames sem atualização
    """
    # Remove estados de pose antigos
    old_pose_states = [
        tid for tid, state in pose_motion_states.items()
        if frame_count - state.last_seen_frame > max_frames_inactive
    ]
    for tid in old_pose_states:
        del pose_motion_states[tid]

    # Remove action states órfãos
    old_action_states = [
        tid for tid in action_states.keys()
        if tid not in pose_motion_states
    ]
    for tid in old_action_states:
        del action_states[tid]
