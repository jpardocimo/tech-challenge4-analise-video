"""
Sistema de Análise de Vídeo com Detecção de Pessoas, Emoções, Atividades e Anomalias

Utiliza YOLO (pose + objetos), DeepFace (emotion + age/gender), InsightFace e MediaPipe
"""

import cv2
import collections
import os

# Importações de configuração
from config.settings import (
    INPUT_VIDEO,
    OUTPUT_VIDEO,
    OUTPUT_REPORT,
    OUTPUT_LOG_CSV,
    OUTPUT_DIR,
    ACTIONS_CONFIG_PATH,
    SKIP_FRAMES,
    LOG_EVERY_N_FRAMES,
    LOG_FLUSH_INTERVAL,
    POSE_MOTION_WINDOW,
    POSE_MOTION_K,
    POSE_MOTION_K_SPATIAL,
    ANOMALY_COOLDOWN_FRAMES,
)

# Importações de modelos
from models.model_loader import load_models

# Importações do core (detecção + análise)
from core.object_detector import detect_objects
from core.person_detector import detect_people_with_tracking
from core.face_detector import detect_faces
from core.activity_analysis import ActivityClassifier
from core.anomaly_detector import process_pose_anomalies, cleanup_old_states
from core.body_face_matcher import match_faces_to_bodies

# Importações de visualização
from visualization.renderer import render_frame

# Importações de reporting
from reporting.logger import VideoAnalysisLogger
from reporting.report_generator import generate_final_report


def process_video():
    """
    Loop principal de processamento do vídeo

    Fases:
    1. Carrega modelos de IA
    2. Inicializa vídeo e logger
    3. Loop frame-by-frame:
       - Detecção (pessoas, faces, objetos)
       - Análise (pose, emoção, atividade, anomalias)
       - Renderização (UI overlay)
       - Logging
    4. Gera relatório final
    """

    # =============================================================================
    # INICIALIZAÇÃO
    # =============================================================================

    # Cria pasta de saída se não existir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Estatísticas globais
    stats = {
        "pessoas_unicas": set(),
        "emocoes": collections.defaultdict(int),
        "atividades": collections.defaultdict(int),
        "anomalias_total": 0,
        "frames_analisados": 0
    }

    # Estados de rastreamento
    pose_motion_states = {}  # track_id -> PoseMotionState
    action_states = {}       # track_id -> ActionState

    # Carrega modelos
    yolo_pose_model, yolo_object_model, face_analysis_model = load_models()

    # Carrega rule engine de atividades
    activity_classifier = ActivityClassifier(ACTIONS_CONFIG_PATH)

    # Abre vídeo
    video_capture = cv2.VideoCapture(INPUT_VIDEO)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Cria video writer
    video_writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (frame_width, frame_height)
    )

    print(f"🎬 Iniciando: {frame_width}x{frame_height} @ {fps}fps")
    print(f"📹 Total de frames: {total_frames}")

    # Inicializa logger
    logger = VideoAnalysisLogger(OUTPUT_LOG_CSV)

    # Cache de emoções
    emotion_cache = {}

    # Persistência de estado entre frames (para frames que não processam IA)
    last_render_data = []
    last_anomalies_list = []

    frame_count = 0

    # =============================================================================
    # LOOP PRINCIPAL
    # =============================================================================

    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            frame_count += 1
            stats['frames_analisados'] += 1

            # Progresso
            if frame_count % 10 == 0:
                print(f"Frame {frame_count}/{total_frames}")

            # ================================================================
            # DETECÇÃO E ANÁLISE (com skip frames para otimização)
            # ================================================================

            should_process_ai = (frame_count % SKIP_FRAMES == 0)

            if should_process_ai:
                # Detecta pessoas com tracking
                bodies = detect_people_with_tracking(frame, yolo_pose_model)

                # Detecta objetos
                objects = detect_objects(frame, yolo_object_model)

                # Detecta faces
                faces = detect_faces(frame, face_analysis_model)

                # Associa faces aos corpos e analisa tudo
                render_data, anomalies_list = match_faces_to_bodies(
                    frame, bodies, faces, emotion_cache, frame_count,
                    objects, activity_classifier, action_states, stats
                )

                # Salva para frames intermediários
                last_render_data = render_data
                last_anomalies_list = anomalies_list
            else:
                # Reusa dados do frame anterior
                render_data = last_render_data
                anomalies_list = list(last_anomalies_list)
                objects = []

            # ================================================================
            # ANÁLISE DE MOVIMENTO POR POSE (detecção de movimentos abruptos)
            # ================================================================

            if should_process_ai and render_data:
                delta_time = float(SKIP_FRAMES) / float(max(1, fps))

                pose_anomalies = process_pose_anomalies(
                    render_data=render_data,
                    pose_motion_states=pose_motion_states,
                    frame_count=frame_count,
                    delta_time=delta_time,
                    fps=fps,
                    pose_motion_window=POSE_MOTION_WINDOW,
                    pose_motion_k_geometric=POSE_MOTION_K,
                    pose_motion_k_spatial=POSE_MOTION_K_SPATIAL,
                    anomaly_cooldown_frames=ANOMALY_COOLDOWN_FRAMES
                )

                # Adiciona anomalias detectadas
                for anomaly_msg in pose_anomalies:
                    if anomaly_msg not in anomalies_list:
                        anomalies_list.append(anomaly_msg)
                        stats["anomalias_total"] += 1

            # ================================================================
            # LOGGING
            # ================================================================

            logger.log_frame_if_needed(
                frame_count=frame_count,
                fps=fps,
                render_data=render_data,
                anomalies=anomalies_list,
                detected_objects=objects if should_process_ai else [],
                log_interval=LOG_EVERY_N_FRAMES
            )
            logger.flush_if_needed(frame_count, LOG_FLUSH_INTERVAL)

            # ================================================================
            # RENDERIZAÇÃO
            # ================================================================

            render_frame(
                frame=frame,
                render_data=render_data,
                anomalies_list=anomalies_list,
                objects=objects,
                should_process_ai=should_process_ai,
                frame_width=frame_width
            )

            # ================================================================
            # LIMPEZA DE ESTADOS ANTIGOS
            # ================================================================

            if frame_count % 30 == 0:
                cleanup_old_states(pose_motion_states, action_states, frame_count)

            # ================================================================
            # OUTPUT
            # ================================================================

            cv2.imshow('Final Hibrido', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            video_writer.write(frame)

    except KeyboardInterrupt:
        print("\n⚠️ Interrompido pelo usuário")

    # =============================================================================
    # FINALIZAÇÃO
    # =============================================================================

    # Fecha recursos
    logger.close()
    video_capture.release()
    video_writer.release()
    cv2.destroyAllWindows()

    # Gera relatório final
    generate_final_report(stats, OUTPUT_REPORT)

    # Mensagens finais
    print(f"\n📝 Log salvo em: {OUTPUT_LOG_CSV}")
    print(f"🎞️ Vídeo salvo em: {OUTPUT_VIDEO}")
    print(f"📄 Relatório salvo em: {OUTPUT_REPORT}")
    print(f"\n✅ Processamento finalizado!")
    print(f"   - Frames processados: {stats['frames_analisados']}")
    print(f"   - Pessoas únicas: {len(stats['pessoas_unicas'])}")
    print(f"   - Anomalias detectadas: {stats['anomalias_total']}")


if __name__ == "__main__":
    process_video()
