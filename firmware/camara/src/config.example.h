// Copiar a config.h en esta misma carpeta (src/) y llenar los valores.
#pragma once

// Red. Debe ser la misma red que la de la PC y la pluma.
#define WIFI_SSID "mi-red"
#define WIFI_PASSWORD "mi-clave"
#define HOSTNAME "camara"  // Queda como camara.local.
#define OTA_PASSWORD "cambiar-esta-clave"
#define STREAM_PORT 81

// Imagen. Para reconocer a ~1 m, XGA deja la cara en ~150 px con el lente normal; la CNN
// usa 96x96, así que basta con que la cara mida unos 80 px. Con lente gran angular sube a
// FRAMESIZE_HD o acércate. También se cambia sin cargar firmware: /control?var=framesize&val=13
// (SVGA 11, XGA 12, HD 13, SXGA 14, UXGA 15).
#define FRAME_SIZE FRAMESIZE_XGA
#define JPEG_QUALITY 10  // De 4 a 63. Menor es mejor calidad y más datos; 10 a 12 va bien.
#define MAX_FPS 15       // El reconocimiento pide 5 frames consistentes: 15 fps sobran.

// Según cómo quede montada la cámara.
#define FLIP_VERTICAL false
#define MIRROR_HORIZONTAL false

// Pines. -1 significa deshabilitado. Los del sensor son los de la AI Thinker y están en main.cpp.
#define FLASH_LED_PIN 4
#define STATUS_LED_PIN 33  // Activo en bajo.

// 1 imprime una línea de estado por segundo en el monitor serie.
#define DEBUG_SERIAL 1
