"""
Módulo para gerenciamento de estatísticas de análise de vídeo

Responsabilidades:
- Encapsular estrutura de estatísticas
- Fornecer métodos para atualizar estatísticas
- Fornecer métodos para consultar estatísticas
"""

from collections import defaultdict
from typing import Dict, Set


class StatsManager:
    """
    Gerenciador centralizado de estatísticas de análise de vídeo
    """

    def __init__(self):
        """Inicializa estruturas de estatísticas"""
        self.pessoas_unicas: Set[int] = set()
        self.emocoes: Dict[str, int] = defaultdict(int)
        self.atividades: Dict[str, int] = defaultdict(int)
        self.anomalias_total: int = 0
        self.frames_analisados: int = 0

    def add_pessoa(self, pessoa_id: int):
        """
        Adiciona uma pessoa única ao registro

        Args:
            pessoa_id: ID da pessoa detectada
        """
        if pessoa_id is not None:
            self.pessoas_unicas.add(pessoa_id)

    def add_emocao(self, emocao: str):
        """
        Registra uma emoção detectada

        Args:
            emocao: Nome da emoção
        """
        if emocao:
            self.emocoes[emocao] += 1

    def add_atividade(self, atividade: str):
        """
        Registra uma atividade detectada

        Args:
            atividade: Nome da atividade
        """
        if atividade:
            self.atividades[atividade] += 1

    def add_anomalia(self):
        """Incrementa contador de anomalias"""
        self.anomalias_total += 1

    def increment_frame(self):
        """Incrementa contador de frames analisados"""
        self.frames_analisados += 1

    def get_summary(self) -> Dict:
        """
        Retorna resumo completo das estatísticas

        Returns:
            Dicionário com todas as estatísticas:
            {
                'frames_analisados': int,
                'pessoas_unicas': set,
                'anomalias_total': int,
                'emocoes': dict,
                'atividades': dict
            }
        """
        return {
            'frames_analisados': self.frames_analisados,
            'pessoas_unicas': self.pessoas_unicas,
            'anomalias_total': self.anomalias_total,
            'emocoes': dict(self.emocoes),
            'atividades': dict(self.atividades)
        }
