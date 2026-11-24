# Licencias y privacidad

MIT del archivo LICENSE se aplica al código propio, cuyos titulares están
confirmados como los dos autores del proyecto. No convierte automáticamente
datasets, fotos, pesos biométricos o terceros en material MIT redistribuible.

YuNet se distribuye una sola vez en los recursos del paquete con su licencia
MIT original, copyright/aviso del autor y manifiesto de procedencia. El commit,
URL y SHA256 exactos figuran en resources/yunet_provenance.json. Se leyó y
conservó la licencia upstream: el contacto del autor es atribución requerida
de terceros, no un participante del proyecto. No se descargan modelos en runtime.

R02/R03-B fueron entrenados con personas reales. No hay autorización explícita
documentada para distribución pública de esos pesos: quedan FUERA de esta versión
pública y sus rutas locales están ignoradas por Git. Una entrega privada o un
video consentido no implican autorización pública. Se conserva una importación
local comprobada para personas autorizadas, sin publicar nombres, fuentes o fotos.

No se incluye VGGFace2 ni muestras externas; no se ha establecido permiso para
redistribuir esa copia. El usuario debe aportar fuentes legales para su propio
entrenamiento. Revisar consentimiento, conservación y eliminación de datos
locales. Una revocación impide volver a entrenar/evaluar con el manifiesto previo;
no elimina automáticamente información ya aprendida por un checkpoint.

No usar el prototipo como seguridad física. No existe detección de vida y puede
ser vulnerable a fotografías/pantallas, incluso con score alto.
