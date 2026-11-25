# Protocolo de la pluma

Fuente de verdad para `firmware/pluma/src/protocol.h` y `src/face_access/barrier.py`.
Viene del protocolo del carrito FPV: UDP, little endian y CRC8 con polinomio 0x07 y valor
inicial 0x00, calculado sobre todos los bytes anteriores al CRC.

## Paquetes

1. **Comando** (PC a pluma, puerto 4211, 10 Hz). 6 bytes, formato struct de Python `"<BBHBB"`:
   - magic uint8 = 0xB0
   - version uint8 = 1
   - seq uint16: sube en cada envío y da la vuelta después de 65535
   - open uint8: 1 abierta, 0 cerrada
   - crc8 uint8

2. **Telemetría** (pluma a PC, 5 Hz). 7 bytes, formato `"<BBHbBB"`:
   - magic uint8 = 0xB1
   - version uint8 = 1
   - last_seq uint16: último seq aceptado
   - rssi int8 en dBm
   - flags uint8: bit 0 failsafe activo, bit 1 el objetivo es "abierta", bit 2 el servo ya llegó al objetivo
   - crc8 uint8

## Reglas

- La PC manda el **estado deseado** en cada paquete, no un evento "abrir". Perder paquetes
  no cambia nada: el siguiente trae el mismo estado.
- La pluma responde a la IP y al puerto de origen del último comando válido.
- Descarta paquetes con magic, versión, tamaño o CRC incorrectos.
- Un comando es nuevo si `(int16)(seq - last_seq) > 0`. Si pasaron más de 1000 ms sin
  comandos válidos acepta cualquier seq, porque la PC pudo reiniciarse.
- **Failsafe:** sin un comando válido en `FAILSAFE_MS` (500 ms), la pluma se cierra.
  También se cierra al iniciar una carga OTA.
- El servo va hacia el objetivo a `SERVO_SPEED_US_PER_S`, no de golpe.

El CRC8 no es autenticación: cualquiera en la misma red puede mandar "abrir". Sirve para
un proyecto escolar en una red propia, no para una instalación real.
