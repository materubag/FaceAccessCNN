# FaceAccessCNN

Reconocimiento facial con una CNN entrenada desde cero, detección YuNet
y simulación temporal de apertura de puerta. Aplicación local de webcam
con flujo de adquisición, revisión, entrenamiento y calibración.

## Tecnologías

Python3.12–3.14, PyTorch, OpenCV, NumPy, scikit-learn y Matplotlib.
YuNet localiza rostros y cinco landmarks; la CNN propia clasifica identidades.
No utiliza embeddings ni clasificadores de identidad preentrenados.

## Funcionalidades

- Detección y ROI cuadrada con controles de calidad.
- Grises, mediana3×3 y tensor1×96×96 en[0,1].
- Rechazo mediante clase desconocida, score y margen top-2.
- Cinco confirmaciones y 1,5s de detección continua; apertura2s y cooldown5s desde autorización.
- Espejo solo visual; diagnóstico y selección de modelo.
- Selección de cámara y backend de captura en Windows (`auto`, `dshow`, `msmf`).
- Captura consentida, revisión humana, splits por sesión e identidades externas.
- Entrenamiento desde cero y calibración exclusivamente con validación.

## Instalación

Desde la raíz del proyecto, crear un entorno nuevo:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
# CPU: no requiere NVIDIA ni CUDA.
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e .
```

Activar `.venv` para usar los comandos siguientes con `python`.
Para GPU, instalar una wheel PyTorch compatible con el controlador/hardware
desde el selector oficial: https://pytorch.org/get-started/locally/.
No se descarga ningún modelo de identidad silenciosamente.
YuNet se incluye una sola vez como recurso del paquete, con licencia/procedencia.

## Demo

Comprobar el pipeline sin cámara ni datos privados:

```powershell
python -m face_access self-test
python -m face_access demo --preview-only --camera 0
```

Los pesos sensibles R02/R03-B **no se distribuyen públicamente**: no existe
autorización pública documentada. Para usarlos, obtener los artefactos por un
canal autorizado e importarlos de forma local, o entrenar un modelo propio.
Los comandos de demo con identidad requieren ese paso; no funcionan con pesos
inexistentes ni se sustituyen por predicciones ficticias.

```powershell
python -m face_access models list
python -m face_access models import --name r03b --checkpoint ruta/al/checkpoint.pt --calibration ruta/a/calibration.json
python -m face_access demo --model r03b --check
python -m face_access demo --model r03b
python -m face_access demo --model r02 --camera 1 --no-mirror --debug
python -m face_access demo --preview-only --camera 1 --camera-backend dshow
```

R03-B es el candidato experimental predeterminado; R02 es la alternativa oficial.
Cada modelo conserva su calibración. La importación comprueba hashes y contrato,
elimina metadatos innecesarios y deja los archivos en `models/`, ignorado por Git.
Q sale, D alterna diagnóstico y M alterna espejo visual.
`--events runs/demo.jsonl` registra transiciones sin fotos/videos.

## Comenzar a entrenar

Capturar cada identidad en sesiones independientes y revisar los candidatos:

```powershell
python -m face_access capture --person persona_01 --session sesion_01 --consent --independent-session
python -m face_access review --person persona_01
python -m face_access prepare --config configs/train.yaml
python -m face_access train --config configs/train.yaml
python -m face_access evaluate --checkpoint runs/face_model/best.pt --manifest data/manifest.csv --output runs/face_model/val
python -m face_access demo --checkpoint runs/face_model/best.pt --calibration runs/face_model/val/calibration.json
```

La preparación solo funciona después de completar identidades/sesiones,
desconocidos legales disjuntos y aprobaciones humanas. Añadir una carpeta o
capturar `persona_04` **no añade una identidad instantáneamente**: actualizar
clases, reconstruir manifiesto, entrenar y recalibrar. Véase el flujo completo.

## Documentación y pruebas

- [Captura, datos, entrenamiento y calibración](docs/training.md).
- [Resultados y limitaciones](docs/results.md).
- [Licencias, privacidad y procedencia](docs/licenses.md).
- [Verificación y equivalencia](docs/verification.md).

```powershell
python -m pip install -e ".[dev]"
python -m pytest -q
python -m face_access --help
```

Los tests habituales usan entradas sintéticas; no necesitan cámara ni biometría.
Los `.yaml` son documentos JSON compatibles con YAML1.2, sin dependencia PyYAML.

## Licencia y límites

Código propio bajo [MIT](LICENSE), copyright2026 de sus autores.
La licencia propia no se extiende a datos, pesos de identidad o terceros.
YuNet conserva su aviso MIT; los pesos biométricos requieren permiso específico.
Sin detección de vida: fotos/pantallas pueden vulnerarlo. No controla cerraduras
reales ni es apto para seguridad física. No publicar capturas o pesos sensibles.
