# Arquitetura do Sistema

## Fluxo de Processamento 

```
┌─────────────────────────────────────────────────────────────────┐
│                         main.py                                  │
│                    (Orquestrador Principal)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────────┐
        │         1. INICIALIZAÇÃO                  │
        │  - Carrega modelos (model_loader.py)     │
        │  - Inicializa logger (logger.py)         │
        │  - Abre vídeo                            │
        └──────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────────┐
        │      2. LOOP FRAME-BY-FRAME              │
        └──────────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                            │
        ▼                                            ▼
┌──────────────────┐                    ┌──────────────────────┐
│   DETECÇÃO       │                    │   ANÁLISE            │
│                  │                    │                      │
│ person_detector  │───────────────────▶│ pose_analyzer        │
│ face_detector    │                    │ activity_classifier  │
│ object_detector  │                    │ anomaly_detector     │
│ emotion_detector │                    │ body_face_matcher    │
└──────────────────┘                    └──────────────────────┘
        │                                            │
        │                                            │
        └─────────────────────┬─────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────────┐
        │         3. RENDERIZAÇÃO                   │
        │  - Desenha skeleton (renderer.py)        │
        │  - Desenha bboxes                        │
        │  - Desenha labels                        │
        │  - Desenha anomalias                     │
        └──────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────────┐
        │         4. LOGGING & OUTPUT               │
        │  - Grava frame no vídeo                  │
        │  - Log em CSV (logger.py)                │
        │  - Atualiza estatísticas                 │
        └──────────────────────────────────────────┘
                              │
                              ▼
        ┌──────────────────────────────────────────┐
        │         5. FINALIZAÇÃO                    │
        │  - Fecha recursos                        │
        │  - Gera relatório (report_generator.py)  │
        └──────────────────────────────────────────┘
```



## Fluxo Detalhado de Análise do Vídeo/Frame

```
Frame de Vídeo
      │
      ├──▶ person_detector.py ──▶ Lista de pessoas (bbox + keypoints + track_id)
      │                                              │
      ├──▶ face_detector.py ──▶ Lista de faces ────┤
      │                          (bbox + embedding)  │
      └──▶ object_detector.py ──▶ Lista de objetos ─┤
                                                     │
                                                     ▼
                            body_face_matcher.py (ORQUESTRADOR)
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
                    ▼                ▼                ▼
            pose_analyzer    face_identity   emotion_detector
                    │                │                │
                    │                │                │
                    └────────────────┼────────────────┘
                                     │
                                     ▼
                         activity_classifier
                                     │
                                     ▼
                          anomaly_detector (movimento)
                                     │
                                     ▼
                         Lista de render_data
                         (pessoas + análises completas)
```
