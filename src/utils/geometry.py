"""
Funções utilitárias para cálculos geométricos e manipulação de keypoints
"""

import numpy as np


def calculate_distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
    """
    Calcula distância euclidiana entre dois pontos

    Args:
        point_a: Ponto A [x, y]
        point_b: Ponto B [x, y]

    Returns:
        Distância euclidiana
    """
    return float(np.linalg.norm(point_a - point_b))


def calculate_angle(point_a: np.ndarray, point_b: np.ndarray, point_c: np.ndarray) -> float:
    """
    Calcula ângulo ABC em graus (B é o vértice)

    Args:
        point_a: Ponto A [x, y]
        point_b: Ponto B (vértice) [x, y]
        point_c: Ponto C [x, y]

    Returns:
        Ângulo em graus ou None se os pontos são colineares
    """
    vector_ba = point_a - point_b
    vector_bc = point_c - point_b

    norm_ba = np.linalg.norm(vector_ba)
    norm_bc = np.linalg.norm(vector_bc)

    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return None

    cosine_angle = float(np.dot(vector_ba, vector_bc) / (norm_ba * norm_bc))
    cosine_angle = max(-1.0, min(1.0, cosine_angle))

    return float(np.degrees(np.arccos(cosine_angle)))


def get_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calcula similaridade de cosseno entre dois embeddings

    Args:
        embedding1: Primeiro embedding
        embedding2: Segundo embedding

    Returns:
        Similaridade de cosseno (0-1)
    """
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
