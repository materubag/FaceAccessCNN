// Copiar a config.h en esta misma carpeta (src/) y llenar los valores.
#pragma once

// Red. Debe ser la misma red que la de la PC y la ESP32-CAM.
#define WIFI_SSID "mi-red"
#define WIFI_PASSWORD "mi-clave"
#define HOSTNAME "pluma"  // Queda como pluma.local.
#define OTA_PASSWORD "cambiar-esta-clave"
#define UDP_PORT 4211

// Pines. -1 significa deshabilitado.
#define SERVO_PIN 13       // Cable naranja del SG90. Evita 0, 2, 12 y 15: son pines de arranque.
#define STATUS_LED_PIN 2   // LED azul de la placa.

// Recorrido del SG90, en microsegundos de pulso. Unos 500 us son 0° y unos 2400 us son 180°.
// Ajusta hasta que la pluma quede horizontal al cerrar y vertical al abrir.
#define SERVO_CLOSED_US 500
#define SERVO_OPEN_US 1450
#define SERVO_SPEED_US_PER_S 2000  // 950 us de recorrido en ~0.5 s. Más bajo es más lento.

// Sin un comando válido en este tiempo, la pluma se cierra.
#define FAILSAFE_MS 500
#define TELEMETRY_PERIOD_MS 200

// 1 imprime una línea de estado por segundo en el monitor serie.
#define DEBUG_SERIAL 0
