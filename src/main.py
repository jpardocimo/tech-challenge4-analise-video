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
    ANOMALY_COOLDOWN_FRAMES
)

# Importações de modelos
from models.model_loader import load_models

# Importações de detecção
from detection.object_detector import detect_objects
from detection.person_detector import detect_people_with_tracking
from detection.face_detector import detect_faces

# Importações de análise
from analysis.pose_analyzer import extract_pose_features, calculate_pose_motion_score
from analysis.activity_classifier import ActivityClassifier
from analysis.anomaly_detector import PoseMotionState
from analysis.body_face_matcher import match_faces_to_bodies

# Importações de visualização
from visualization.renderer import (
    draw_objects,
    draw_person_ui,
    draw_anomaly_banner,
    draw_text_buffer
)

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

            can_update_pose_motion = should_process_ai
            delta_time = float(SKIP_FRAMES) / float(max(1, fps))

            if can_update_pose_motion and render_data:
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
                            window=POSE_MOTION_WINDOW,
                            k=POSE_MOTION_K
                        )
                        pose_motion_states[track_id] = pose_state

                    # Calcula score de movimento
                    motion_score = calculate_pose_motion_score(
                        pose_state.previous_features,
                        features,
                        delta_time
                    )

                    pose_state.previous_features = features
                    pose_state.last_seen_frame = frame_count

                    if motion_score is None:
                        item["pose_dbg"] = "poseScore=-- (no-score)"
                        continue

                    # Detecta anomalia
                    is_anomaly, mean, threshold = pose_state.detector.update(motion_score)

                    # Anti-jitter: precisa de confirmação
                    if mean is None or threshold is None:
                        pose_state.spike_count = 0
                        item["pose_dbg"] = f"poseScore={motion_score:.3f} (warmup)"
                        continue

                    delta = motion_score - threshold

                    if is_anomaly:
                        pose_state.spike_count += 1
                    else:
                        pose_state.spike_count = 0

                    is_anomaly_final = (pose_state.spike_count >= 1) or (delta > 10.0)
                    item["pose_dbg"] = (
                        f"poseScore={motion_score:.3f} thr={threshold:.3f} "
                        f"Δ={delta:.3f} spikes={pose_state.spike_count}"
                    )

                    # Dispara anomalia com cooldown
                    if is_anomaly_final and (frame_count - pose_state.last_anomaly_frame) > ANOMALY_COOLDOWN_FRAMES:
                        pose_state.last_anomaly_frame = frame_count
                        pose_state.spike_count = 0

                        anomaly_msg = f"MOVIMENTO ABRUPTO (POSE T{track_id})"
                        if anomaly_msg not in item["anomalies"]:
                            item["anomalies"].append(anomaly_msg)

                        if anomaly_msg not in anomalies_list:
                            anomalies_list.append(anomaly_msg)
                            stats["anomalias_total"] += 1

            # ================================================================
            # LOGGING
            # ================================================================

            if frame_count % LOG_EVERY_N_FRAMES == 0:
                timestamp_ms = int((frame_count * 1000) / fps)
                logger.log_frame(
                    frame_count,
                    timestamp_ms,
                    render_data,
                    anomalies_list,
                    objects if should_process_ai else []
                )

            # Flush periódico do buffer de log
            if frame_count % LOG_FLUSH_INTERVAL == 0:
                logger.flush()

            # ================================================================
            # RENDERIZAÇÃO
            # ================================================================

            # Desenha objetos detectados
            if should_process_ai:
                draw_objects(frame, objects)

            text_buffer = []

            # Banner de anomalia
            has_anomaly = len(anomalies_list) > 0
            if has_anomaly:
                anomaly_text = draw_anomaly_banner(frame, anomalies_list, frame_width)
                if anomaly_text:
                    text_buffer.append(anomaly_text)

            # Desenha UI de cada pessoa
            for item in render_data:
                draw_person_ui(frame, text_buffer, item)

            # Desenha todos os textos
            draw_text_buffer(frame, text_buffer)

            # ================================================================
            # LIMPEZA DE ESTADOS ANTIGOS (otimizado: a cada 30 frames)
            # ================================================================

            if frame_count % 30 == 0:
                # Remove estados de pose antigos
                old_pose_states = [
                    tid for tid, state in pose_motion_states.items()
                    if frame_count - state.last_seen_frame > 90
                ]
                for tid in old_pose_states:
                    del pose_motion_states[tid]

                # Remove action states sem pose state correspondente
                old_action_states = [
                    tid for tid in action_states.keys()
                    if tid not in pose_motion_states
                ]
                for tid in old_action_states:
                    del action_states[tid]

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
