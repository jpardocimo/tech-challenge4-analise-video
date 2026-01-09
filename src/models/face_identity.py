"""
Módulo para gestão de identidades faciais
"""

import numpy as np
from utils.geometry import get_cosine_similarity
from config.settings import FACE_SIMILARITY_THRESHOLD

# Banco de dados global de faces (embedding por ID)
_global_face_database = {}
_next_face_id = 1


def resolve_face_identity(embedding: np.ndarray) -> int:
    """
    Resolve identidade facial usando similaridade de embeddings

    Compara o embedding com o banco de dados de faces conhecidas.
    Se encontrar match (similaridade > threshold), retorna o ID existente.
    Caso contrário, cria nova identidade.

    Args:
        embedding: Embedding facial (vetor 512-d do InsightFace)

    Returns:
        ID da face (inteiro único)
    """
    global _global_face_database, _next_face_id

    best_id = None
    best_similarity = 0

    # Busca face mais similar no banco
    for face_id, db_embedding in _global_face_database.items():
        similarity = get_cosine_similarity(embedding, db_embedding)
        if similarity > best_similarity:
            best_similarity = similarity
            best_id = face_id

    # Se similaridade é alta o suficiente, é a mesma pessoa
    if best_similarity > FACE_SIMILARITY_THRESHOLD:
        # Atualiza embedding com média móvel (90% antigo + 10% novo)
        _global_face_database[best_id] = (
            _global_face_database[best_id] * 0.9 + embedding * 0.1
        )
        return best_id
    else:
        # Nova identidade
        new_id = _next_face_id
        _next_face_id += 1
        _global_face_database[new_id] = embedding
        return new_id


def get_total_identities() -> int:
    """
    Retorna o número total de identidades únicas no banco

    Returns:
        Número de identidades únicas
    """
    return len(_global_face_database)
