// Formato de los paquetes UDP de la pluma. La especificación está en docs/protocolo-pluma.md.
#pragma once

#include <Arduino.h>

const uint8_t MAGIC_COMMAND = 0xB0;
const uint8_t MAGIC_TELEMETRY = 0xB1;
const uint8_t PROTOCOL_VERSION = 1;

const uint8_t FLAG_FAILSAFE = 1 << 0;   // Sin comandos recientes: la pluma se cierra.
const uint8_t FLAG_OPENING = 1 << 1;    // El objetivo actual es "abierta".
const uint8_t FLAG_AT_TARGET = 1 << 2;  // El servo ya llegó al objetivo.

struct __attribute__((packed)) CommandPacket {
  uint8_t magic;
  uint8_t version;
  uint16_t seq;
  uint8_t open;
  uint8_t crc;
};

struct __attribute__((packed)) TelemetryPacket {
  uint8_t magic;
  uint8_t version;
  uint16_t lastSeq;
  int8_t rssi;
  uint8_t flags;
  uint8_t crc;
};

static_assert(sizeof(CommandPacket) == 6, "CommandPacket debe medir 6 bytes");
static_assert(sizeof(TelemetryPacket) == 7, "TelemetryPacket debe medir 7 bytes");

// CRC-8 con polinomio 0x07 y valor inicial 0x00.
uint8_t crc8(const uint8_t *data, size_t len);

// Comprueba tamaño, versión y CRC. El magic lo revisa quien decide el tipo de paquete.
bool isValidPacket(const uint8_t *data, int size, size_t expectedSize);
