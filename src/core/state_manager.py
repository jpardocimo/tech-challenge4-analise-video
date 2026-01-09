"""
Módulo para gerenciamento de estados de tracking

Responsabilidades:
- Gerenciar estados de pose (PoseMotionState)
- Gerenciar estados de ação (ActionState)
- Fornecer interface unificada para acesso aos estados
- Limpeza periódica de estados antigos
"""

from typing import Dict, Optional
from core.anomaly_detector import cleanup_old_states


class StateManager:
    """
    Gerenciador centralizado de estados de tracking
    """

    def __init__(self):
        """Inicializa dicionários de estados"""
        self.pose_motion_states: Dict = {}
        self.action_states: Dict = {}

    def get_pose_state(self, track_id):
        """
        Retorna estado de movimento de pose para um track_id

        Args:
            track_id: ID de tracking

        Returns:
            PoseMotionState ou None
        """
        return self.pose_motion_states.get(track_id)

    def set_pose_state(self, track_id, state):
        """
        Define estado de movimento de pose para um track_id

        Args:
            track_id: ID de tracking
            state: PoseMotionState
        """
        self.pose_motion_states[track_id] = state

    def get_action_state(self, track_id):
        """
        Retorna estado de ação para um track_id

        Args:
            track_id: ID de tracking

        Returns:
            ActionState ou None
        """
        return self.action_states.get(track_id)

    def set_action_state(self, track_id, state):
        """
        Define estado de ação para um track_id

        Args:
            track_id: ID de tracking
            state: ActionState
        """
        self.action_states[track_id] = state

    def cleanup_old_states(self, frame_count: int, max_inactive: int = 90):
        """
        Remove estados antigos de rastreamento

        Args:
            frame_count: Frame atual
            max_inactive: Máximo de frames sem atualização
        """
        cleanup_old_states(
            self.pose_motion_states,
            self.action_states,
            frame_count,
            max_inactive
        )
