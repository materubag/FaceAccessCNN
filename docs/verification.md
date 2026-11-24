# Verificación de esta versión

Verificado el 24 de noviembre de 2026 en Windows 11, Python 3.14.3. Entorno virtual
temporal nuevo, fuera de este repositorio: torch2.14.1+cpu, NumPy2.5.5,
OpenCV contrib5.1.0.95, scikit-learn1.9.1, Matplotlib3.11.2 y pytest9.2.0.

## Pruebas realizadas

- Instalación editable con `pip install -e ".[dev]"` y dependencias compatibles.
- Ayuda de todos los subcomandos y punto de entrada `python -m face_access`.
- `self-test` desde un directorio ajeno al proyecto, sin GPU/cámara/datos privados.
- Construcción e instalación no editable de una wheel. Repetición de self-test
  con acceso al repositorio original bloqueado: detector y recursos resueltos
  desde el paquete instalado, no mediante rutas del workspace académico.
- `python -m pytest -q`: **24 pruebas aprobadas**, en cinco archivos más conftest.
  Cubren tensor/grises/normalización, clases dinámicas, geometría, detección
  nula/múltiple/ambigua, espejo visual, umbral/margen, calibración ajena o ausente,
  temporalidad/seguridad, consentimiento previo a escritura, muestreo de video
  simulado60FPS, revisión vigente/revocada, fugas/duplicados y guardado/carga.
  La quinta agrupación conserva una regresión de la demo: espejo visual, cámara
  liberada y ningún uso de la ROI del fotograma anterior tras una detección inválida.
- Recorrido real por CLI, con archivos sintéticos temporales:
  `prepare → train (una época CPU) → evaluate (val) → demo --check`.
  La suite comprueba también el marcador de test con datos exclusivamente
  sintéticos. No se ejecutó ninguna nueva inferencia sobre test histórico.
- R02/R03-B se importaron temporalmente mediante la misma API de la CLI, fuera
  del repositorio público. Los tensores coincidieron exactamente con sus originales;
  diferencia máxima de logits0 y probabilidades idénticas sobre la misma ROI
  sintética CPU. Calibraciones byte a byte intactas y ambos `demo --check` correctos.
- En un fotograma sintético compartido, ROI y tensor coincidieron con el pipeline
  original. Una secuencia temporal completa produjo los mismos estados,
  contadores, aperturas y cooldown que el controlador original con tres autorizados.

Los pesos usados para estas comprobaciones NO se incluyen ni quedan en models/.
Esto no es una autorización de publicación, una evaluación de accuracy ni una
prueba física de webcam. No se modificaron los modelos/resultados originales. Se comprobaron 2.502
archivos hist?ricos protegidos mediante hashes, sin diferencias.

## Reutilización y adaptaciones

| Componente del prototipo | Módulo de esta versión | Adaptación |
|---|---|---|
| Detección/geometría/ROI/deduplicación | detection | YuNet único como recurso; sin Haar ni rutas del workspace. |
| Transformación base y nitidez/hash | preprocessing | Misma base; errores legibles para archivos vacíos. |
| FaceCNN | model | Misma arquitectura; num_classes configurable,4 por defecto. |
| DoorState | door | Semántica original; número de autorizados configurable y pausa segura para K clases. |
| HUD y espejo visual | display/demo | Vista heredada, modelo visible; Q/D/M; sin decisiones de verdad suministradas por el operador. |
| Afines moderadas | augmentation | Mismos rangos/probabilidades, solo train. |
| Captura/revisión/manifiestos | acquisition/dataset | Extraídos controles útiles; eventos humanos y fuentes/sesiones sin administración histórica fija. |
| Adam/AMP/early stopping/métricas | training/evaluation/metrics | Misma estrategia; clases genéricas y calibración enlazada, sin calibrar test. |

El recorte de una caja completamente fuera de imagen ahora falla también si se
llama directamente a square_crop. El pipeline ya rechazaba esos casos antes de
recortar; las ROI válidas no cambian. No se modificaron parámetros del detector,
arquitecturas de referencia, normalización, filtros ni umbrales seleccionados.

## Pendientes y alcance

La webcam física de esta reorganización no se abrió: pruebas de cámara con mocks,
no una nueva comprobación presencial. Probar permisos, selección de cámara y
rendimiento en el equipo receptor. GPU en esta reorganización, otros sistemas
operativos y otra computadora no se verificaron físicamente.

Las pruebas presenciales anteriores comunicadas por el usuario corresponden al
prototipo y no se convierten en una métrica nueva de esta publicación. R03-B
sigue sin test independiente nuevo. Los pesos de identidad permanecen excluidos
por falta de consentimiento público documentado. No hay API web, infraestructura
de entrega académica ni descargas automáticas de modelos sensibles.
