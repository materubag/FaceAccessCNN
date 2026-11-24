# Modelos locales, fuera de Git

No hay consentimiento público documentado para publicar los pesos de las tres
personas del proyecto. Por ello este repositorio NO incluye r02.pt ni r03b.pt,
sus calibraciones ni archivos biométricos. No confundir captura consentida o
entrega académica con autorización para un repositorio público.

Se mantienen las dos opciones mediante `face-access models import`. La persona
autorizada debe proporcionar el checkpoint completo original o su exportación
de inferencia seleccionada y la calibración correspondiente. Solo se aceptan
los hashes congelados R02/R03-B. No se descargan de internet ni se sustituyen por
modelos preentrenados de identidad. Los archivos importados se guardan aquí,
están ignorados por Git y conservan tensores/umbrales/margen exactos.

Sin acceso privado, se puede usar `demo --preview-only`, `self-test` y el flujo
completo para entrenar identidades propias. Un modelo nuevo requiere validación
y calibración antes de poder autorizar en la demo.

YuNet está incluido una sola vez en `face_access/resources/yunet.onnx`, como
recurso instalable independiente del directorio de ejecución.
                                               