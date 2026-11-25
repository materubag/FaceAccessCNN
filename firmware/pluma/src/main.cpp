// Firmware de la pluma: ESP32 DevKit V1 con un servo SG90.
// Recibe de la PC el estado deseado (abierta o cerrada) por UDP. El protocolo está en
// docs/protocolo-pluma.md. Basado en el firmware del carrito FPV: mismo WiFi, OTA, CRC8 y seq.

#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <ArduinoOTA.h>

#include "protocol.h"

// Va después de los headers del core para que sus macros no los afecten.
#if __has_include("config.h")
#include "config.h"
#else
#error "Falta config.h: copia src/config.example.h a src/config.h y llena los valores"
#endif

static_assert(SERVO_CLOSED_US >= 400 && SERVO_CLOSED_US <= 2600, "SERVO_CLOSED_US fuera del rango del SG90");
static_assert(SERVO_OPEN_US >= 400 && SERVO_OPEN_US <= 2600, "SERVO_OPEN_US fuera del rango del SG90");
static_assert(SERVO_SPEED_US_PER_S > 0, "SERVO_SPEED_US_PER_S debe ser positivo");

const uint8_t SERVO_CHANNEL = 0;
const uint32_t SERVO_PWM_FREQ = 50;
const uint8_t SERVO_PWM_BITS = 16;
const uint32_t SERVO_PERIOD_US = 20000;
// Un periodo del PWM: el servo no ve cambios más rápidos que esto.
const uint32_t SERVO_UPDATE_MS = 20;

const uint32_t SEQ_RESYNC_MS = 1000;
const uint32_t WIFI_RETRY_MS = 3000;
const uint32_t LED_FAST_MS = 100;
const uint32_t LED_SLOW_MS = 500;
const uint32_t DEBUG_PERIOD_MS = 1000;

// ---------- Estado ----------
WiFiUDP udp;

bool wifiConnected = false;
bool otaStarted = false;
uint32_t lastWifiAttemptMs = 0;

// commandReceived también indica que peerIp y peerPort ya son válidos.
bool commandReceived = false;
uint16_t lastSeq = 0;
uint32_t lastCommandMs = 0;
IPAddress peerIp;
uint16_t peerPort = 0;

bool openRequested = false;
bool failsafeActive = true;
int servoUs = SERVO_CLOSED_US;
uint32_t lastServoMs = 0;

uint32_t lastTelemetryMs = 0;
uint32_t lastDebugMs = 0;
uint32_t packetsReceived = 0;
uint32_t packetsDropped = 0;
uint32_t loopsCount = 0;
uint32_t maxWriteUs = 0;

// ---------- Servo ----------
void writeServo(int pulseUs) {
  uint32_t startUs = micros();
  ledcWrite(SERVO_PIN, (uint32_t)pulseUs * 65535 / SERVO_PERIOD_US);
  maxWriteUs = max(maxWriteUs, micros() - startUs);
}

int targetUs() {
  return openRequested && !failsafeActive ? SERVO_OPEN_US : SERVO_CLOSED_US;
}

// Cierra de golpe, sin la rampa. Se usa al iniciar una carga OTA, que bloquea el loop.
void closeNow() {
  openRequested = false;
  servoUs = SERVO_CLOSED_US;
  writeServo(servoUs);
}

// Mueve el servo poco a poco hacia el objetivo: se ve como una pluma y evita el pico de
// corriente de un salto completo, que puede reiniciar el ESP32 si el servo va a VIN.
// El paso sale del tiempo transcurrido y no de las vueltas del loop: en el ESP32 clásico,
// ledcWrite espera a que el PWM de 50 Hz tome el valor y tarda hasta ~40 ms.
void updateServo(uint32_t now) {
  failsafeActive = !wifiConnected || !commandReceived || now - lastCommandMs > FAILSAFE_MS;

  uint32_t elapsedMs = now - lastServoMs;
  if (elapsedMs < SERVO_UPDATE_MS) return;
  lastServoMs = now;

  int target = targetUs();
  if (servoUs == target) return;
  int step = max(1, (int)((uint32_t)SERVO_SPEED_US_PER_S * min(elapsedMs, (uint32_t)1000) / 1000));
  servoUs = servoUs < target ? min(servoUs + step, target) : max(servoUs - step, target);
  writeServo(servoUs);
}

// ---------- UDP ----------
bool isNewSeq(uint16_t seq, uint32_t now) {
  if (!commandReceived || now - lastCommandMs > SEQ_RESYNC_MS) return true;
  return (int16_t)(seq - lastSeq) > 0;
}

// Solo un comando aceptado renueva lastCommandMs. Si un seq viejo también lo renovara,
// una PC reiniciada con seq bajo nunca llegaría a los SEQ_RESYNC_MS y quedaría ignorada.
bool acceptCommand(const uint8_t *data, uint32_t now) {
  CommandPacket command;
  memcpy(&command, data, sizeof(command));
  if (!isNewSeq(command.seq, now)) return false;

  openRequested = command.open != 0;
  lastSeq = command.seq;
  lastCommandMs = now;
  commandReceived = true;
  peerIp = udp.remoteIP();
  peerPort = udp.remotePort();
  return true;
}

// Vacía la cola UDP. Cada comando trae el estado completo, así que basta el último.
void receivePackets(uint32_t now) {
  int size;
  while ((size = udp.parsePacket()) > 0) {
    packetsReceived++;
    bool accepted = false;
    uint8_t data[sizeof(CommandPacket)];

    if ((size_t)size == sizeof(data)) {
      udp.read(data, size);
      if (data[0] == MAGIC_COMMAND && isValidPacket(data, size, sizeof(CommandPacket))) {
        accepted = acceptCommand(data, now);
      }
    }

    // parsePacket devuelve 0 mientras quede un paquete a medio leer, como uno muy largo.
    udp.clear();
    if (!accepted) packetsDropped++;
  }
}

void sendTelemetry(uint32_t now) {
  if (!commandReceived || now - lastTelemetryMs < TELEMETRY_PERIOD_MS) return;
  lastTelemetryMs = now;

  int target = targetUs();
  TelemetryPacket telemetry;
  telemetry.magic = MAGIC_TELEMETRY;
  telemetry.version = PROTOCOL_VERSION;
  telemetry.lastSeq = lastSeq;
  telemetry.rssi = (int8_t)WiFi.RSSI();
  telemetry.flags = (failsafeActive ? FLAG_FAILSAFE : 0)
                    | (target == SERVO_OPEN_US ? FLAG_OPENING : 0)
                    | (servoUs == target ? FLAG_AT_TARGET : 0);
  telemetry.crc = crc8((const uint8_t *)&telemetry, sizeof(telemetry) - 1);

  udp.beginPacket(peerIp, peerPort);
  udp.write((const uint8_t *)&telemetry, sizeof(telemetry));
  udp.endPacket();
}

// ---------- WiFi y OTA ----------
void onWifiConnected() {
  udp.begin(UDP_PORT);

  // ArduinoOTA.begin() también inicia mDNS, que anuncia HOSTNAME.local.
  if (!otaStarted) {
    ArduinoOTA.begin();
    otaStarted = true;
  }

  // Siempre se imprime: si Windows no resuelve pluma.local, esta IP va en la config de la PC.
  Serial.printf("WiFi conectado: %s.local = %s, UDP %d\n", HOSTNAME, WiFi.localIP().toString().c_str(), UDP_PORT);
}

void updateWifi(uint32_t now) {
  bool connected = WiFi.status() == WL_CONNECTED;
  if (connected && !wifiConnected) onWifiConnected();
  wifiConnected = connected;
  if (connected) return;

  // Ya asociado y esperando IP: un reintento cortaría el DHCP en curso.
  if (!WiFi.STA.connected() && now - lastWifiAttemptMs >= WIFI_RETRY_MS) {
    lastWifiAttemptMs = now;
    WiFi.reconnect();
  }
}

// ---------- Diagnóstico de WiFi ----------
// El evento llega desde la tarea del WiFi; el motivo se imprime en loop.
volatile uint8_t disconnectReason = 0;

void onWifiDisconnected(WiFiEvent_t, WiFiEventInfo_t info) {
  disconnectReason = info.wifi_sta_disconnected.reason;
}

void reportWifiProblem() {
  static uint8_t lastReported = 0;
  uint8_t reason = disconnectReason;
  if (reason == 0 || reason == lastReported) return;
  lastReported = reason;
  Serial.printf("WiFi no conecta a \"%s\": %s (%u)\n", WIFI_SSID,
                WiFi.disconnectReasonName((wifi_err_reason_t)reason), reason);
}

// Antes de WiFi.begin: muestra qué redes de 2.4 GHz alcanza la placa.
void printNetworks() {
  int count = WiFi.scanNetworks();
  Serial.printf("Redes de 2.4 GHz visibles: %d\n", count);
  for (int i = 0; i < count; i++) {
    Serial.printf("  %-32s %4d dBm  canal %2d%s\n", WiFi.SSID(i).c_str(), WiFi.RSSI(i), WiFi.channel(i),
                  WiFi.SSID(i) == WIFI_SSID ? "  <- configurada" : "");
  }
  WiFi.scanDelete();
}

// ---------- Indicadores ----------
// Rápido: sin WiFi. Lento: sin comandos de la PC (pluma cerrada). Fijo: la PC está conectada.
void updateStatusLed(uint32_t now) {
#if STATUS_LED_PIN >= 0
  bool on = true;
  if (!wifiConnected) {
    on = (now / LED_FAST_MS) % 2 == 0;
  } else if (failsafeActive) {
    on = (now / LED_SLOW_MS) % 2 == 0;
  }
  digitalWrite(STATUS_LED_PIN, on ? HIGH : LOW);
#else
  (void)now;
#endif
}

#if DEBUG_SERIAL
void printDebug(uint32_t now) {
  if (now - lastDebugMs < DEBUG_PERIOD_MS) return;
  lastDebugMs = now;

  Serial.printf("rx=%lu drop=%lu abierta=%d failsafe=%d servo=%dus rssi=%d loops=%lu ledc_max=%luus\n",
                (unsigned long)packetsReceived, (unsigned long)packetsDropped, openRequested, failsafeActive, servoUs,
                wifiConnected ? (int)WiFi.RSSI() : 0, (unsigned long)loopsCount, (unsigned long)maxWriteUs);
  packetsReceived = 0;
  packetsDropped = 0;
  loopsCount = 0;
  maxWriteUs = 0;
}
#endif

// ---------- Arduino ----------
void setup() {
  Serial.begin(115200);

  // Lo primero: la pluma cerrada.
  if (!ledcAttachChannel(SERVO_PIN, SERVO_PWM_FREQ, SERVO_PWM_BITS, SERVO_CHANNEL)) {
    Serial.println("Error al configurar el PWM del servo");
  }
  closeNow();

#if STATUS_LED_PIN >= 0
  pinMode(STATUS_LED_PIN, OUTPUT);
#endif

  ArduinoOTA.setHostname(HOSTNAME);
  ArduinoOTA.setPassword(OTA_PASSWORD);
  ArduinoOTA.onStart(closeNow);

  // setHostname debe ir antes de mode(). La reconexión la lleva updateWifi cada 3 s.
  WiFi.setHostname(HOSTNAME);
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.setAutoReconnect(false);
  printNetworks();
  WiFi.onEvent(onWifiDisconnected, ARDUINO_EVENT_WIFI_STA_DISCONNECTED);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  lastWifiAttemptMs = millis();
  Serial.printf("Conectando a %s...\n", WIFI_SSID);
}

void loop() {
  // handle() bloquea durante toda la subida OTA; onStart ya cerró la pluma.
  if (wifiConnected) ArduinoOTA.handle();

  uint32_t now = millis();
  updateWifi(now);
  if (!wifiConnected) reportWifiProblem();
  if (wifiConnected) {
    receivePackets(now);
    sendTelemetry(now);
  }

  updateServo(now);
  updateStatusLed(now);

#if DEBUG_SERIAL
  loopsCount++;
  printDebug(now);
#endif

  delay(1);
}
