# Modelos locales, fuera de Git

No hay consentimiento público documentado para publicar los pesos de las tres
personas del proyecto. Por ello este repositorio NO incluye r02.pt ni r03b.pt,
sus calibraciones ni archivos biométricos. No confundir captura consentida o
entrega académica con autorización para un repositorio público.

Se mantienen las dos opciones mediante `face-access models import`. La persona
autorizada debe proporcionar el checkpoint completo original o su exportación
de inferencia seleccionada y la calibración correspondiente. Solo se aceptan
los hashes congelados para los nombres R02/R03-B. No se descargan de internet ni se sustituyen por
modelos preentrenados de identidad. Los archivos importados se guardan aquí,
están ignorados por Git y conservan tensores/umbrales/margen exactos.

Sin acceso privado, se puede usar `demo --preview-only`, `self-test` y el flujo
completo para entrenar identidades propias. Un modelo nuevo requiere validación
y calibración antes de poder autorizar en la demo.

YuNet está incluido una sola vez en `face_access/resources/yunet.onnx`, como
recurso instalable independiente del directorio de ejecución.

## Experimentos locales

Otros checkpoints propios pueden convertirse con `models import --name r06`
(o un nombre nuevo), `--checkpoint`, `--calibration` y, si existe,
`--safety-attestation`. Se validan pesos, etiquetas, contrato YuNet y vinculación
de la calibración; no se cambian los registros congelados de R02/R03-B.
Sin evidencia válida y satisfactoria, la importación conserva solo diagnóstico:
no abre la puerta ni conecta con la pluma, aunque se indique `--barrier`.

Los archivos importados se usan explícitamente, sin cambiar el modelo predeterminado:

```powershell
python -m face_access demo --checkpoint models/r06.pt --calibration models/r06_calibration.json --camera "http://camara.local:81/stream" --debug
```

R06 y R07 de desarrollo permanecen bloqueados por fallos de reconocimiento;
su disponibilidad local no significa que estén aprobados para autorizar.
Estos pesos y calibraciones siguen excluidos de Git. No se publica consentimiento
para distribuirlos y no deben añadirse con `git add -f`.
