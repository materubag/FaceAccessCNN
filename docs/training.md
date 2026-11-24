# Entrenar con identidades propias

## Datos y consentimiento

Usar identificadores anónimos y fuentes propias o disponibles legalmente.
`--consent` confirma permisos de uso local de imágenes; no autoriza publicación
de fotografías ni pesos. `--independent-session` es una declaración del operador
de que la grabación es independiente; el programa no puede comprobarlo físicamente.
No reutilizar fragmentos de un video entre splits. No subir data/runs/models.

```text
data/captures/
  authorized/persona_01/sesion_01/<run>/  # train
  authorized/persona_01/sesion_02/<run>/  # val
  authorized/persona_01/sesion_03/<run>/  # test reservado
  authorized/persona_02/...
  authorized/persona_03/...
  unknown/train/externo_01/sesion_01/<run>/
  unknown/train/externo_02/sesion_01/<run>/
  unknown/val/externo_03/sesion_01/<run>/
  unknown/val/externo_04/sesion_01/<run>/
  unknown/test/externo_05/sesion_01/<run>/
  unknown/test/externo_06/sesion_01/<run>/
```

## Adquirir y revisar

```powershell
python -m face_access capture --person persona_01 --session sesion_01 --source 0 --consent --independent-session
python -m face_access capture --person persona_01 --session sesion_02 --source ruta/video_independiente.mp4 --consent --independent-session --no-preview
python -m face_access capture --person persona_01 --session sesion_03 --source ruta/otro_video.mp4 --consent --independent-session --no-preview
python -m face_access review --person persona_01
```

Repetir para las otras personas. `--source` también admite una carpeta de fotos
legítimas. Webcam/video se muestrean a3FPS por defecto (permitido2–5), con límite
configurable `--limit`. Los originales no se copian ni se modifican. Se guardan
recortes96×96 y procedencia hash/frame. Solo rostros únicos candidate pasan los
controles congelados de confianza, geometría, landmarks, tamaño, nitidez y
duplicados; todos siguen pendientes de aprobación. Capturas ambiguas no se
guardan como aceptadas. No se amplían criterios para aumentar cantidades.

Para desconocidos, usar otras personas/fuentes permitidas y etiquetar su split:

```powershell
python -m face_access capture --person externo_01 --session sesion_01 --unknown --split train --source ruta/imagenes_legales --consent --independent-session --no-preview
python -m face_access review --person externo_01
```

Repetir con al menos dos identidades diferentes por split según configuración.
No usar una misma identidad externa en train/val/test ni llamar desconocida a una
persona autorizada. No seleccionar imágenes usando resultados de la CNN de test.

La revisión OpenCV muestra crops ampliados: A acepta, R rechaza, P deja pendiente,
J/L navegan y Q sale. Cada decisión se guarda inmediatamente en eventos append-only
con hash. Reabrir recupera la última decisión; aceptar nunca es automático.
Una nueva decisión puede revocar otra. Rehacer manifiesto con una ruta versionada
si hay cambios; auditoría previa de entrenamiento/evaluación bloquea revocaciones.

## Preparar y entrenar

Editar `configs/train.yaml`: authorized, sesiones, mínimos y output. Los defaults
exigen10 imágenes aprobadas por identidad/split y dos identidades desconocidas por
split. El número de salidas se calcula como autorizados+1 (Desconocido al final).

```powershell
python -m face_access prepare --config configs/train.yaml
python -m face_access train --config configs/train.yaml --device cpu
# GPU compatible disponible:
python -m face_access train --config configs/train.yaml --device cuda
```

Elegir UNA ejecución y una carpeta nueva para cada experimento; no ejecutar los
dos comandos anteriores con el mismo output. `auto` es el dispositivo por defecto.
No hay pesos iniciales externos ni de R02/R03-B. Se verifica resolución, perfil,
hashes, revisión, fuente, sesiones, clases, identidades desconocidas y duplicados
exactos/casi idénticos entre splits. Se conserva test sin clasificar.

CNN Flatten predeterminada: tres bloques32/64/128, kernel3,padding1,stride1,
pooling2; Dense64,Dropout0,3,K salidas. GAP opcional cambiando model.head.
Adam0,001, batch32, weight decay0,0001, semilla42, hasta50 épocas, paciencia7.
CrossEntropyLoss con pesos inversos de clase; AMP en CUDA. Mejor best.pt por
val_loss, history.csv/json y curvas. No se guardan rutas personales en el checkpoint.

Solo train tiene espejo p0,5; afines p0,7 (±8°,escala0,92–1,08,traslación±4 %),
brillo0,85–1,15 y contraste p0,3 (0,9–1,1). RNG de aumentos separado de Dropout.
Cambiar augmentation.name a original conserva espejo/brillo previos, o none para
desactivar. Val/inferencia siempre usan la base original sin aumentos.

## Validar, calibrar y usar

```powershell
python -m face_access evaluate --checkpoint runs/face_model/best.pt --manifest data/manifest.csv --output runs/face_model/val --split val
python -m face_access demo --checkpoint runs/face_model/best.pt --calibration runs/face_model/val/calibration.json --check
python -m face_access demo --checkpoint runs/face_model/best.pt --calibration runs/face_model/val/calibration.json
```

Se calibran score global y margen usando exclusivamente val; objetivo FAR0,01 y
margen0,15 por defecto. Los desconocidos de val no aparecieron en train. Una muestra
pequeña puede obligar a rechazar muchos autorizados; no relajar para aparentar éxito.
La calibración incluye SHA del checkpoint/manifiesto y split val. No puede
intercambiarse entre redes. `--smoke` entrena una época de depuración y marca el
checkpoint: jamás se admite para autorizar ni evaluar un test final.

## Test, una vez tras congelar decisiones

```powershell
python -m face_access evaluate --checkpoint runs/face_model/best.pt --manifest data/manifest.csv --output runs/face_model/test --split test --calibration runs/face_model/val/calibration.json --confirm-frozen
```

El comando valida todo antes de predecir, crea un marcador exclusivo y no permite
repetición automática, incluso si una pasada se interrumpe. No se usa test para
elegir arquitectura, aumentos o umbral. La métrica por imagen y forward por batch
no son tasas de apertura temporal ni FPS de webcam.

## Añadir persona_04

Capturar sus sesiones con `capture --person persona_04 --session ...` no cambia
el modelo existente. Añadir el identificador a authorized o sustituir otro,
preparar un manifiesto nuevo, elegir output nuevo, entrenar de cero y recalibrar.
La cabeza densa se reconstruye a K salidas; las calibraciones/modelos viejos no
son intercambiables. Un test ya observado no vuelve a estar reservado: definir
nuevas sesiones independientes para cualquier nueva afirmación de generalización.
