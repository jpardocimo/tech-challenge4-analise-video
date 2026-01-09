"""
Sistema de Análise de Vídeo com Detecção de Pessoas, Emoções, Atividades e Anomalias

Utiliza YOLO (pose + objetos), DeepFace (emotion), InsightFace

Responsabilidades do main.py:
- Inicializar sistema e modelos
- Orquestrar loop principal
- Coordenar entrada → processamento → saída
"""

import os
import cv2

# Configuração
from config.settings import (
    INPUT_VIDEO, OUTPUT_VIDEO, OUTPUT_REPORT, OUTPUT_LOG_CSV, OUTPUT_DIR,
    ACTIONS_CONFIG_PATH, SKIP_FRAMES, LOG_EVERY_N_FRAMES, LOG_FLUSH_INTERVAL,
)

# Modelos
from models.model_loader import load_models

# Detecção
from core.object_detector import detect_objects
from core.person_detector import detect_people_with_tracking
from core.face_detector import detect_faces
from core.activity_analysis import ActivityClassifier
from core.body_face_matcher import match_faces_to_bodies
from core.emotion_detector import EmotionManager

# Importar wrapper de anomalias
from core.anomaly_detector import detect_and_merge_pose_anomalies

# Gerenciadores
from reporting.stats_manager import StatsManager
from core.state_manager import StateManager
from utils.video_utils import VideoReader, VideoWriter, FrameProcessor, ProgressTracker

# Visualização e reporting
from visualization.renderer import render_frame
from reporting.logger import VideoAnalysisLogger
from reporting.report_generator import generate_final_report


def process_video():
    """Loop principal de processamento do vídeo"""

    # =============================================================================
    # INICIALIZAÇÃO
    # =============================================================================
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Modelos IA
    yolo_pose_model, yolo_object_model, face_analysis_model = load_models()
    activity_classifier = ActivityClassifier(ACTIONS_CONFIG_PATH)

    # Gerenciadores
    stats_manager = StatsManager()
    state_manager = StateManager()
    emotion_manager = EmotionManager()
    frame_processor = FrameProcessor(SKIP_FRAMES)

    # I/O de vídeo
    video_reader = VideoReader(INPUT_VIDEO)
    video_props = video_reader.get_properties()
    video_writer = VideoWriter(
        OUTPUT_VIDEO, video_props['fps'],
        video_props['width'], video_props['height']
    )

    # UI e logging
    progress = ProgressTracker(video_props['total_frames'])
    logger = VideoAnalysisLogger(OUTPUT_LOG_CSV)

    print(f"🎬 Iniciando: {video_props['width']}x{video_props['height']} @ {video_props['fps']}fps")
    print(f"📹 Total de frames: {video_props['total_frames']}")

    # =============================================================================
    # LOOP PRINCIPAL
    # =============================================================================

    frame_count = 0

    try:
        while True:
            ret, frame = video_reader.read_frame()
            if not ret:
                break

            frame_count += 1
            stats_manager.increment_frame()
            progress.update(frame_count)

            # Processa IA ou reusa cache
            if frame_processor.should_process_ai(frame_count):
                # Detecção
                bodies = detect_people_with_tracking(frame, yolo_pose_model)
                objects = detect_objects(frame, yolo_object_model)
                faces = detect_faces(frame, face_analysis_model)

                # Análise (pose + emoção + atividade)
                render_data, anomalies_list = match_faces_to_bodies(
                    frame, bodies, faces, frame_count, objects,
                    activity_classifier, state_manager, stats_manager, emotion_manager
                )

                # Anomalias de movimento (encapsulado no anomaly_detector)
                detect_and_merge_pose_anomalies(
                    render_data, anomalies_list, state_manager.pose_motion_states,
                    frame_count, video_props['fps'], SKIP_FRAMES, stats_manager
                )

                frame_processor.cache_results(render_data, anomalies_list, objects)
            else:
                render_data, anomalies_list, objects = frame_processor.get_cached_results()

            # Renderização
            render_frame(
                frame, render_data, anomalies_list, objects,
                frame_processor.should_process_ai(frame_count),
                video_props['width']
            )

            # Logging
            logger.log_frame_if_needed(
                frame_count, video_props['fps'], render_data,
                anomalies_list, objects, LOG_EVERY_N_FRAMES
            )
            logger.flush_if_needed(frame_count, LOG_FLUSH_INTERVAL)

            # Limpeza periódica de estados antigos
            if frame_count % 30 == 0:
                state_manager.cleanup_old_states(frame_count)

            # Output
            cv2.imshow('Final Hibrido', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            video_writer.write_frame(frame)

    except KeyboardInterrupt:
        print("\n⚠️ Interrompido pelo usuário")

    # =============================================================================
    # FINALIZAÇÃO
    # =============================================================================

    logger.close()
    video_reader.release()
    video_writer.release()
    cv2.destroyAllWindows()

    # Relatório
    stats_summary = stats_manager.get_summary()
    generate_final_report(stats_summary, OUTPUT_REPORT)
    progress.print_summary(stats_summary, OUTPUT_LOG_CSV, OUTPUT_VIDEO, OUTPUT_REPORT)


if __name__ == "__main__":
    process_video()
