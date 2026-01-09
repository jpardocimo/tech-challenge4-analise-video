"""
Utilitários para processamento de vídeo

Consolida:
- I/O de vídeo (VideoReader, VideoWriter)
- Processamento de frames (FrameProcessor)
- Tracking de progresso (ProgressTracker)
"""

import cv2
from typing import Tuple, Dict, Optional, List


# =============================================================================
# I/O DE VÍDEO
# =============================================================================

class VideoReader:
    """Leitor de vídeo que encapsula cv2.VideoCapture"""

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.capture = cv2.VideoCapture(video_path)
        if not self.capture.isOpened():
            raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")

        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.capture.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def read_frame(self) -> Tuple[bool, Optional[any]]:
        """Lê próximo frame do vídeo"""
        return self.capture.read()

    def release(self):
        """Libera recursos do vídeo"""
        self.capture.release()

    def get_properties(self) -> Dict[str, int]:
        """Retorna propriedades do vídeo"""
        return {
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'total_frames': self.total_frames
        }


class VideoWriter:
    """Escritor de vídeo que encapsula cv2.VideoWriter"""

    def __init__(self, output_path: str, fps: int, width: int, height: int, codec: str = 'mp4v'):
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not self.writer.isOpened():
            raise RuntimeError(f"Não foi possível criar o vídeo: {output_path}")

    def write_frame(self, frame):
        """Escreve um frame no vídeo"""
        self.writer.write(frame)

    def release(self):
        """Libera recursos do escritor"""
        self.writer.release()


# =============================================================================
# PROCESSAMENTO DE FRAMES
# =============================================================================

class FrameProcessor:
    """Gerenciador de processamento de frames com skip frames"""

    def __init__(self, skip_frames: int):
        self.skip_frames = skip_frames
        self.last_render_data = []
        self.last_anomalies_list = []
        self.last_objects = []

    def should_process_ai(self, frame_count: int) -> bool:
        """Determina se deve processar IA no frame atual"""
        return frame_count % self.skip_frames == 0

    def cache_results(self, render_data: List, anomalies_list: List, objects: List):
        """Armazena resultados do processamento para reuso"""
        self.last_render_data = render_data
        self.last_anomalies_list = anomalies_list
        self.last_objects = objects

    def get_cached_results(self) -> Tuple[List, List, List]:
        """Retorna resultados em cache do último processamento"""
        return (
            self.last_render_data,
            list(self.last_anomalies_list),
            []  # Objetos não são reutilizados em frames cached
        )


# =============================================================================
# TRACKING DE PROGRESSO
# =============================================================================

class ProgressTracker:
    """Rastreador de progresso de processamento de vídeo"""

    def __init__(self, total_frames: int, update_interval: int = 10):
        self.total_frames = total_frames
        self.update_interval = update_interval

    def update(self, frame_count: int):
        """Atualiza exibição de progresso"""
        if frame_count % self.update_interval == 0:
            print(f"Frame {frame_count}/{self.total_frames}")

    def print_summary(self, stats_summary: Dict, output_log_csv: str, output_video: str, output_report: str):
        """Exibe resumo final de processamento"""
        print(f"\n📝 Log salvo em: {output_log_csv}")
        print(f"🎞️ Vídeo salvo em: {output_video}")
        print(f"📄 Relatório salvo em: {output_report}")
        print(f"\n✅ Processamento finalizado!")
        print(f"   - Frames processados: {stats_summary['frames_analisados']}")
        print(f"   - Pessoas únicas: {len(stats_summary['pessoas_unicas'])}")
        print(f"   - Anomalias detectadas: {stats_summary['anomalias_total']}")
