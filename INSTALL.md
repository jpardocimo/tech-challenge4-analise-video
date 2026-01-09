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

### Pré-requisitos

1. **Python 3.10 ou 3.11**
   ```cmd
   python --version
   # Deve mostrar: Python 3.10.x ou 3.11.x
   ```

2. **Microsoft Visual C++ 14.0+** (necessário para algumas bibliotecas)

   Baixe e instale:
   - [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - Durante instalação, marque: **"Desktop development with C++"**

3. **Visual C++ Redistributable** (para runtime)

   Baixe e instale:
   - [VC++ Redistributable x64](https://aka.ms/vs/17/release/vc_redist.x64.exe)

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

# 6. Verifique instalação
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

# 7. Verifique CUDA
python -c "import torch; print('CUDA disponível:', torch.cuda.is_available())"
```

### O que é instalado

- **tensorflow**: TensorFlow padrão para Windows
- **torch**: PyTorch CPU ou CUDA (dependendo da opção escolhida)
- **insightface**: Requer Visual C++ (daí os pré-requisitos)
- Todas as outras dependências

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
1. Instale [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. Durante instalação, marque: "Desktop development with C++"
3. Reinicie o computador
4. Tente instalar novamente

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

#### Problema: "InsightFace failed to install"
**Solução:**
```cmd
# Instale dependências de build primeiro
pip install cmake
pip install insightface --no-cache-dir
```

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
✅ Pillow                      OK

============================================================

ACELERAÇÃO GPU:
------------------------------------------------------------
PyTorch versão: 2.1.0
✅ Apple MPS (Metal) disponível  # ou CUDA no Windows

TensorFlow versão: 2.15.0
✅ TensorFlow GPU: 1 dispositivo(s)

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
