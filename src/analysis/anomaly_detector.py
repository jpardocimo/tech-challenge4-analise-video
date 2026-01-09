"""
Módulo para detecção de anomalias de movimento
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
    Rastreia estado de movimento de pose para detecção de movimentos abruptos

    Mantém histórico de features de pose e detecta spikes de movimento
    usando RollingAnomalyDetector.
    """

    def __init__(self, window: int = 30, k: float = 2.2):
        """
        Args:
            window: Tamanho da janela para detecção
            k: Multiplicador de desvio padrão
        """
        self.detector = RollingAnomalyDetector(window=window, k=k)
        self.previous_features = None
        self.last_seen_frame = -1
        self.last_anomaly_frame = -10
        self.spike_count = 0
