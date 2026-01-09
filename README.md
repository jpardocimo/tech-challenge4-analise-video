# Sistema de Análise de Vídeo - Tech Challenge 4

Sistema de análise de vídeo com detecção de pessoas, emoções, atividades e anomalias usando YOLO, InsightFace e DeepFace.

## Estrutura do Projeto

```
tech-challenge4-analise-video/
├── src/
│   ├── main.py                    # Entry point - execute este arquivo
│   │
│   ├── config/                    # Configurações
│   │   ├── settings.py            # Configurações globais (paths, thresholds)
│   │   ├── colors.py              # Cores da UI (em visualization/)
│   │   └── actions_config.json    # Regras de atividades
│   │
│   ├── models/                    # Gestão de modelos
│   │   ├── model_loader.py        # Carregamento de YOLO, InsightFace
│   │   └── face_identity.py       # Reconhecimento facial
│   │
│   ├── detection/                 # Módulos de detecção
│   │   ├── object_detector.py     # Detecção de objetos (YOLO)
│   │   ├── person_detector.py     # Detecção de pessoas com tracking
│   │   ├── face_detector.py       # Detecção facial (InsightFace)
│   │   └── emotion_detector.py    # Análise de emoções (DeepFace)
│   │
│   ├── analysis/                  # Módulos de análise
│   │   ├── pose_analyzer.py       # Análise de pose e posturas
│   │   ├── activity_classifier.py # Classificação de atividades
│   │   ├── anomaly_detector.py    # Detecção de anomalias
│   │   └── body_face_matcher.py   # Associação face-corpo
│   │
│   ├── visualization/             # Renderização
│   │   ├── renderer.py            # Desenho de UI (skeleton, bboxes, labels)
│   │   └── colors.py              # Definições de cores
│   │
│   ├── reporting/                 # Logging e relatórios
│   │   ├── logger.py              # Logging em CSV
│   │   └── report_generator.py   # Relatórios finais
│   │
│   └── utils/                     # Utilitários
│       └── geometry.py            # Funções geométricas (distância, ângulo)
│
├── ai_models/                     # Modelos de IA (YOLO, etc)
├── output/                        # Saída de vídeos e relatórios
└── venv/                          # Ambiente virtual Python
```

## Como Usar

### Instalação

#### MacBook Pro M1/M2/M3 (Apple Silicon)

```bash
# 1. Crie um ambiente virtual
python3 -m venv venv

# 2. Ative o ambiente virtual
source venv/bin/activate

# 3. Instale as dependências (otimizadas para M1)
pip install --upgrade pip
pip install -r requirements-mac-m1.txt
```

#### Windows 10/11

```bash
# 1. Crie um ambiente virtual
python -m venv venv

# 2. Ative o ambiente virtual
venv\Scripts\activate

# 3. Instale as dependências
pip install --upgrade pip
pip install -r requirements-windows.txt
```

**📝 Nota:** Se você tem GPU NVIDIA no Windows e quer usar aceleração CUDA, veja as instruções no arquivo `requirements-windows.txt`

**Dependências principais:**
- `opencv-python`: Processamento de vídeo
- `ultralytics`: YOLO (detecção de pessoas e objetos)
- `insightface`: Detecção e reconhecimento facial
- `deepface`: Análise de emoções
- `torch`: Backend para YOLO (otimizado por sistema)
- `tensorflow`: Backend para DeepFace (otimizado por sistema)

### Execução

```bash
# Execute o sistema
python src/main.py

# Os outputs serão salvos na pasta output/:
# - output/video_final_YYYYMMDD_HHMMSS.mp4
# - output/relatorio_final_YYYYMMDD_HHMMSS.txt
# - output/overlay_log_YYYYMMDD_HHMMSS.csv
```

### Configuração

Edite `src/config/settings.py` para ajustar:
- Caminhos de vídeo de entrada/saída
- Performance (SKIP_FRAMES, etc)
- Thresholds de detecção
- Parâmetros de anomalias

Edite `src/config/actions_config.json` para configurar regras de atividades.

## Funcionalidades

### Detecção
- **Pessoas**: Tracking persistente com YOLO Pose (ByteTrack)
- **Faces**: Detecção e reconhecimento facial (InsightFace)
- **Emoções**: Análise de emoções com DeepFace
- **Objetos**: Celular, laptop, livro, mesa, cadeira, etc

### Análise
- **Pose**: 17 keypoints COCO, ângulos articulares, posturas (em pé, sentado, deitado)
- **Atividades**: Rule engine configurável (trabalhando, lendo, acenando, etc)
- **Anomalias**: Quedas, movimentos abruptos, instabilidade emocional

### Saídas
- **Vídeo**: Vídeo processado com overlays visuais
- **CSV**: Log detalhado frame-by-frame
- **Relatório**: Estatísticas finais (pessoas, emoções, atividades, anomalias)


## Arquivos Importantes

### Entry Point
- `src/main.py`: Loop principal do sistema

### Configuração
- `src/config/settings.py`: Todas as configurações
- `src/config/actions_config.json`: Regras de atividades

### Core
- `src/analysis/pose_analyzer.py`: Análise de pose (maior módulo)
- `src/analysis/body_face_matcher.py`: Orquestração de análise
- `src/models/model_loader.py`: Carregamento de modelos

## Backup

O arquivo original foi salvo como:
- `analise_final_hibrida_boa_v11-FINAL.py.backup`

## Performance

- `SKIP_FRAMES = 2`: Processa IA a cada 2 frames (ajuste conforme hardware)
- Cache de emoções: Reduz overhead do DeepFace
- Buffer de logging: Minimiza I/O
- ByteTrack: Tracking eficiente de pessoas

## Desenvolvimento

### Adicionar Nova Atividade
1. Edite `src/config/actions_config.json`
2. Adicione regra com condições de pose e objetos
3. Sistema detectará automaticamente

### Adicionar Novo Detector
1. Crie módulo em `src/detection/`
2. Implemente função de detecção
3. Importe e use em `src/main.py`

### Modificar Renderização
- Edite `src/visualization/renderer.py`
- Ajuste cores em `src/visualization/colors.py`

## Licença

FIAP - Pós-graduação - Grupo 117 - Fase 4
