"""
Módulo para logging de dados em CSV
"""

import csv
from typing import List, Dict


class VideoAnalysisLogger:
    """
    Logger para salvar dados de análise de vídeo em CSV

    Mantém buffer interno e faz flush periódico para otimizar I/O
    """

    def __init__(self, csv_path: str):
        """
        Args:
            csv_path: Caminho para o arquivo CSV
        """
        self.csv_path = csv_path
        self.file = open(csv_path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file, delimiter=";")

        # Escreve cabeçalho
        self.writer.writerow([
            "frame", "timestamp_ms", "track_id", "face_id", "tid_used",
            "activity", "emotion", "pose_dbg", "anomalies", "bbox", "objects_detected"
        ])

        self.buffer = []

    def log_frame(
        self,
        frame_count: int,
        timestamp_ms: int,
        render_data: List[Dict],
        anomalies: List[str],
        detected_objects: List[Dict]
    ):
        """
        Registra dados de um frame no buffer

        Args:
            frame_count: Número do frame
            timestamp_ms: Timestamp em milissegundos
            render_data: Dados de renderização das pessoas
            anomalies: Lista de anomalias detectadas
            detected_objects: Lista de objetos detectados
        """
        objects_str = ", ".join([o['label'] for o in detected_objects])

        if not render_data:
            self.buffer.append([
                frame_count, timestamp_ms, "", "", "", "", "", "",
                "|".join(anomalies or []), "", objects_str
            ])
            return

        for item_index, item in enumerate(render_data):
            track_id = item.get("track_id", None)
            face_id = item.get("face_id", None)

            # Calcula ID temporário para uso
            temp_id = track_id
            if temp_id is None:
                temp_id = int(face_id) if face_id is not None else (-1000 - item_index)

            pose_debug = item.get("pose_dbg", "")
            anomalies_str = "|".join(item.get("anomalies", []) or [])
            bbox = item.get("coords", None)

            self.buffer.append([
                frame_count,
                timestamp_ms,
                track_id if track_id is not None else "",
                face_id if face_id is not None else "",
                temp_id,
                item.get("activity", ""),
                item.get("emotion", ""),
                pose_debug,
                anomalies_str,
                f"{bbox}" if bbox is not None else "",
                objects_str
            ])

    def flush(self):
        """
        Escreve todos os logs acumulados no CSV de uma vez
        """
        if self.buffer:
            self.writer.writerows(self.buffer)
            self.buffer.clear()

    def close(self):
        """
        Faz flush final e fecha o arquivo
        """
        self.flush()
        self.file.close()
