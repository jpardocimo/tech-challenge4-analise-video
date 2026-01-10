# Guia de Instalação Detalhado

## Índice
- [MacBook Pro M1/M2/M3](#macbook-pro-m1m2m3-apple-silicon)
- [Windows 10/11](#windows-1011)
- [Problemas Comuns](#problemas-comuns)
- [Verificação da Instalação](#verificação-da-instalação)

---

## MacBook Pro M1/M2/M3 (Apple Silicon)

### Pré-requisitos

1. **Python 3.10 ou 3.11**
   ```bash
   python3 --version
   # Deve mostrar: Python 3.10.x ou 3.11.x
   ```

2. **Xcode Command Line Tools** (necessário para compilar algumas bibliotecas)
   ```bash
   xcode-select --install
   ```

### Instalação Passo a Passo

```bash
# 1. Clone ou navegue até o projeto
cd tech-challenge4-analise-video

# 2. Crie ambiente virtual
python3 -m venv venv

# 3. Ative o ambiente virtual
source venv/bin/activate

# 4. Atualize pip e ferramentas de build
pip install --upgrade pip setuptools wheel

# 5. Instale dependências otimizadas para M1
pip install -r requirements-mac-m1.txt

# 6. Verifique instalação
python -c "import torch; print('PyTorch MPS disponível:', torch.backends.mps.is_available())"
```

### O que é instalado

- **tensorflow-macos**: TensorFlow otimizado para Apple Silicon
- **tensorflow-metal**: Aceleração GPU usando chip M1/M2/M3
- **torch com MPS**: PyTorch com suporte Metal Performance Shaders
- Todas as outras dependências compatíveis

### Desempenho Esperado

- **Com aceleração GPU (MPS/Metal)**: 3-5x mais rápido que CPU
- **Consumo de energia**: Muito eficiente
- **Processamento de vídeo**: ~15-30 FPS dependendo da resolução

---

## Windows 10/11

### ⚠️ PRÉ-REQUISITO OBRIGATÓRIO - LEIA PRIMEIRO!

**ANTES de instalar as dependências Python, você DEVE instalar o Visual Studio Build Tools:**

1. **Baixe o Visual Studio Build Tools:**
   - Acesse: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Baixe "Build Tools for Visual Studio 2022"

2. **Durante a instalação, marque obrigatoriamente:**
   - ✅ "Desenvolvimento para Desktop com C++"
   - ✅ "Ferramentas de Build C++ v143 - VS 2022 C++ x64/x86"

3. **Instale também o Visual C++ Redistributable:**
   - https://aka.ms/vs/17/release/vc_redist.x64.exe

4. **Reinicie o computador após a instalação**

**❌ SEM o Visual Studio Build Tools, a instalação do insightface FALHARÁ!**

---

### Pré-requisitos do Python

1. **Python 3.10 ou 3.11**
   ```cmd
   python --version
   # Deve mostrar: Python 3.10.x ou 3.11.x
   ```

### Instalação Passo a Passo

#### Opção 1: Apenas CPU (mais simples)

```cmd
# 1. Navegue até o projeto
cd tech-challenge4-analise-video

# 2. Crie ambiente virtual
python -m venv venv

# 3. Ative o ambiente virtual
venv\Scripts\activate

# 4. Atualize pip
python -m pip install --upgrade pip setuptools wheel

# 5. Instale dependências
pip install -r requirements-windows.txt

# 6. Baixe modelos do InsightFace (primeira execução)
# Os modelos são baixados automaticamente (~600MB)
python -c "from insightface.app import FaceAnalysis; print('Modelos OK')"

# 7. Verifique instalação
python -c "import torch; print('PyTorch instalado:', torch.__version__)"
```

#### Opção 2: Com GPU NVIDIA (CUDA) - Mais rápido

**Requisitos:**
- Placa de vídeo NVIDIA (GTX 1060 ou superior)
- Drivers NVIDIA atualizados

```cmd
# 1-3. Mesmo que Opção 1

# 4. Atualize pip
python -m pip install --upgrade pip setuptools wheel

# 5. Instale PyTorch com CUDA PRIMEIRO
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 6. Instale resto das dependências
pip install -r requirements-windows.txt

# 7. Baixe modelos do InsightFace
python -c "from insightface.app import FaceAnalysis; print('Modelos OK')"

# 8. Verifique CUDA
python -c "import torch; print('CUDA disponível:', torch.cuda.is_available())"
```

### O que é instalado

- **tensorflow**: TensorFlow padrão para Windows
- **tf-keras**: Compatibilidade com TensorFlow 2.20+
- **torch**: PyTorch CPU ou CUDA (dependendo da opção escolhida)
- **insightface**: Detecção e reconhecimento facial (requer Visual C++)
- **ultralytics**: YOLO11 para pose e detecção de objetos
- **deepface**: Análise de emoções
- **onnxruntime**: Backend para modelos ONNX do InsightFace
- **Modelos do InsightFace**: Baixados automaticamente (~600MB)

### Desempenho Esperado

- **CPU apenas**: ~5-10 FPS
- **GPU NVIDIA (CUDA)**: ~30-60 FPS (depende da GPU)

---

## Problemas Comuns

### MacBook M1

#### Problema: "No module named 'tensorflow'"
```bash
# Solução: Reinstale tensorflow-macos
pip uninstall tensorflow tensorflow-macos tensorflow-metal
pip install tensorflow-macos tensorflow-metal
```

#### Problema: PyTorch não reconhece MPS
```bash
# Solução: Verifique versão do Python
python3 --version  # Deve ser 3.10 ou 3.11

# Reinstale PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio
```

### Windows

#### Problema: "error: Microsoft Visual C++ 14.0 is required"
**Solução:**
1. Instale [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Durante instalação, marque: **"Desktop development with C++"**
3. Marque também: **"Ferramentas de Build C++ v143"**
4. Reinicie o computador
5. Tente instalar novamente

#### Problema: "DLL load failed while importing cv2"
**Solução:**
1. Instale [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)
2. Reinstale OpenCV:
   ```cmd
   pip uninstall opencv-python
   pip install opencv-python
   ```

#### Problema: CUDA não é detectado
**Soluções:**
1. Verifique se tem GPU NVIDIA:
   ```cmd
   nvidia-smi
   ```
2. Atualize drivers NVIDIA
3. Reinstale PyTorch com CUDA:
   ```cmd
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

#### Problema: "No module named 'tf_keras'" com TensorFlow 2.20+
**Solução:**
```cmd
pip install tf-keras
```

#### Problema: InsightFace falha ao carregar modelos
**Solução:**
1. Verifique se os modelos foram baixados:
   ```cmd
   dir %USERPROFILE%\.insightface\models\buffalo_l
   ```
2. Se a pasta estiver vazia ou com arquivo corrompido (9 bytes):
   ```cmd
   del %USERPROFILE%\.insightface\models\buffalo_l\detection.onnx
   ```
3. O sistema recarregará os modelos automaticamente

---

## Verificação da Instalação

### Script de Teste Completo

Crie arquivo `test_install.py`:

```python
#!/usr/bin/env python3
"""Script para verificar instalação das dependências"""

import sys

def test_imports():
    """Testa importação de todas as bibliotecas"""

    print("="*60)
    print("VERIFICANDO INSTALAÇÃO")
    print("="*60)

    modules = {
        'OpenCV': 'cv2',
        'NumPy': 'numpy',
        'PyTorch': 'torch',
        'TorchVision': 'torchvision',
        'Ultralytics (YOLO)': 'ultralytics',
        'InsightFace': 'insightface',
        'ONNX Runtime': 'onnxruntime',
        'DeepFace': 'deepface',
        'TensorFlow': 'tensorflow',
        'tf-keras': 'tf_keras',
        'Pillow': 'PIL'
    }

    failed = []

    for name, module in modules.items():
        try:
            __import__(module)
            print(f"✅ {name:25s} OK")
        except ImportError as e:
            print(f"❌ {name:25s} ERRO: {e}")
            failed.append(name)

    print("\n" + "="*60)

    # Verifica modelos do InsightFace
    print("\nMODELOS INSIGHTFACE:")
    print("-"*60)
    import os
    models_dir = os.path.expanduser('~/.insightface/models/buffalo_l')
    if os.path.exists(models_dir):
        files = os.listdir(models_dir)
        onnx_files = [f for f in files if f.endswith('.onnx')]
        if onnx_files:
            print(f"✅ {len(onnx_files)} modelo(s) encontrado(s):")
            for f in onnx_files:
                size_mb = os.path.getsize(os.path.join(models_dir, f)) / (1024*1024)
                print(f"   - {f} ({size_mb:.1f} MB)")
        else:
            print("⚠️  Nenhum modelo .onnx encontrado")
    else:
        print("⚠️  Pasta de modelos não existe")

    # Verifica aceleração
    print("\nACELERAÇÃO GPU:")
    print("-"*60)

    try:
        import torch
        print(f"PyTorch versão: {torch.__version__}")

        if torch.cuda.is_available():
            print(f"✅ CUDA disponível: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("✅ Apple MPS (Metal) disponível")
        else:
            print("⚠️  Apenas CPU (sem aceleração GPU)")
    except:
        pass

    try:
        import tensorflow as tf
        print(f"\nTensorFlow versão: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ TensorFlow GPU: {len(gpus)} dispositivo(s)")
        else:
            print("⚠️  TensorFlow usando CPU")
    except:
        pass

    print("\n" + "="*60)

    if failed:
        print(f"\n❌ FALHOU: {len(failed)} módulo(s)")
        print("Módulos com problema:", ", ".join(failed))
        return False
    else:
        print("\n✅ SUCESSO! Todas as dependências instaladas corretamente!")
        return True

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
```

**Execute:**
```bash
# Mac/Linux
python test_install.py

# Windows
python test_install.py
```

### Saída Esperada

```
============================================================
VERIFICANDO INSTALAÇÃO
============================================================
✅ OpenCV                      OK
✅ NumPy                       OK
✅ PyTorch                     OK
✅ TorchVision                 OK
✅ Ultralytics (YOLO)          OK
✅ InsightFace                 OK
✅ ONNX Runtime                OK
✅ DeepFace                    OK
✅ TensorFlow                  OK
✅ tf-keras                    OK
✅ Pillow                      OK

============================================================

MODELOS INSIGHTFACE:
------------------------------------------------------------
✅ 6 modelo(s) encontrado(s):
   - 1k3d68.onnx (137.0 MB)
   - 2d106det.onnx (4.8 MB)
   - det_10g.onnx (16.1 MB)
   - genderage.onnx (1.3 MB)
   - w600k_r50.onnx (166.3 MB)

============================================================

ACELERAÇÃO GPU:
------------------------------------------------------------
PyTorch versão: 2.1.0
✅ Apenas CPU (sem aceleração GPU)  # ou CUDA no Windows

TensorFlow versão: 2.20.0
⚠️  TensorFlow usando CPU

============================================================

✅ SUCESSO! Todas as dependências instaladas corretamente!
```

---

## Suporte

Se encontrar problemas não listados aqui:

1. Verifique as issues do GitHub
2. Consulte a documentação oficial:
   - [PyTorch](https://pytorch.org/get-started/locally/)
   - [TensorFlow](https://www.tensorflow.org/install)
   - [InsightFace](https://github.com/deepinsight/insightface)
3. Verifique versão do Python (recomendado: 3.10 ou 3.11)