"""Enlace UDP con la pluma (ESP32 DevKit + servo SG90). Protocolo en docs/protocolo-pluma.md.

Se manda el estado deseado 10 veces por segundo en lugar de un "abre" suelto: un paquete
perdido no importa y, si la PC se cierra o se cuelga, la ESP32 cierra la pluma sola por
failsafe. Basado en el enlace del carrito FPV (mismo CRC8 y seq).
"""
import ipaddress
import socket
import struct
import threading
import time
from dataclasses import dataclass
from .sources import resolve_host

MAGIC_COMMAND = 0xB0
MAGIC_TELEMETRY = 0xB1
PROTOCOL_VERSION = 1
COMMAND = struct.Struct('<BBHBB')
TELEMETRY = struct.Struct('<BBHbBB')
FLAG_FAILSAFE = 1 << 0
FLAG_OPENING = 1 << 1
FLAG_AT_TARGET = 1 << 2

DEFAULT_PORT = 4211
SEND_PERIOD_S = .1
CONNECTED_TIMEOUT_S = 1.
RESOLVE_RETRY_S = 2.
CLOSE_PACKETS = 5


def crc8(data):
    """CRC-8 con polinomio 0x07 y valor inicial 0x00."""
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def encode_command(seq, opened):
    body = COMMAND.pack(MAGIC_COMMAND, PROTOCOL_VERSION, seq & 0xFFFF, 1 if opened else 0, 0)[:-1]
    return body + bytes([crc8(body)])


@dataclass(frozen=True)
class Telemetry:
    last_seq: int
    rssi: int
    flags: int


def decode_telemetry(data):
    """Telemetría válida o None."""
    if len(data) != TELEMETRY.size or data[0] != MAGIC_TELEMETRY or crc8(data[:-1]) != data[-1]:
        return None
    magic, version, last_seq, rssi, flags, _ = TELEMETRY.unpack(data)
    if version != PROTOCOL_VERSION:
        return None
    return Telemetry(last_seq, rssi, flags)


@dataclass(frozen=True)
class BarrierStatus:
    connected: bool = False
    failsafe: bool = True
    opening: bool = False
    at_target: bool = False
    rssi: int | None = None
    ip: str | None = None

    def describe(self):
        if self.ip is None:
            return 'PLUMA: buscando en la red'
        if not self.connected:
            return f'PLUMA: sin respuesta de {self.ip}'
        if self.failsafe:
            return 'PLUMA: failsafe, cerrada'
        movement = '' if self.at_target else (' (abriendo)' if self.opening else ' (cerrando)')
        return f"PLUMA: {'abierta' if self.opening else 'cerrada'}{movement} | {self.rssi} dBm"


def _is_ip(host):
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


class BarrierLink:
    def __init__(self, host, port=DEFAULT_PORT, resolver=resolve_host, period=SEND_PERIOD_S):
        self.host = host
        self.port = int(port)
        self._resolver = resolver
        self._period = period
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._wanted = False
        self._seq = 0
        self._ip = host if _is_ip(host) else None
        self._next_resolve = 0.
        self._last_rx = None
        self._telemetry = None
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Windows: un ICMP "puerto inalcanzable" hace fallar el siguiente recvfrom; se ignora.
        if hasattr(socket, 'SIO_UDP_CONNRESET'):
            self._sock.ioctl(socket.SIO_UDP_CONNRESET, False)
        self._thread = threading.Thread(target=self._run, name='pluma', daemon=True)
        self._thread.start()

    def set_open(self, opened):
        with self._lock:
            self._wanted = bool(opened)

    def status(self):
        with self._lock:
            ip, last_rx, telemetry = self._ip, self._last_rx, self._telemetry
        connected = last_rx is not None and time.monotonic() - last_rx < CONNECTED_TIMEOUT_S
        if not connected or telemetry is None:
            return BarrierStatus(ip=ip)
        return BarrierStatus(connected=True, failsafe=bool(telemetry.flags & FLAG_FAILSAFE),
                             opening=bool(telemetry.flags & FLAG_OPENING), at_target=bool(telemetry.flags & FLAG_AT_TARGET),
                             rssi=telemetry.rssi, ip=ip)

    def close(self):
        """Pide cerrar unas cuantas veces y suelta el socket. El failsafe cubre si no llegan."""
        self.set_open(False)
        self._stop.set()
        self._thread.join(timeout=RESOLVE_RETRY_S + 1)
        with self._lock:
            ip = self._ip
        if ip is not None:
            for _ in range(CLOSE_PACKETS):
                self._send(ip, False)
        self._sock.close()

    def _send(self, ip, opened):
        with self._lock:
            self._seq = (self._seq + 1) & 0xFFFF
            packet = encode_command(self._seq, opened)
        try:
            self._sock.sendto(packet, (ip, self.port))
        except OSError:
            pass  # Red caída: se reintenta en el siguiente periodo.

    def _resolve(self, now):
        # Si la pluma calla un rato puede haber cambiado de IP: se vuelve a buscar.
        with self._lock:
            ip, last_rx = self._ip, self._last_rx
        stale = ip is not None and (last_rx is None or now - last_rx > 3 * CONNECTED_TIMEOUT_S)
        if _is_ip(self.host) or now < self._next_resolve or (ip is not None and not stale):
            return
        self._next_resolve = now + RESOLVE_RETRY_S
        found = self._resolver(self.host)
        if found is not None:
            with self._lock:
                self._ip = found

    def _receive(self, timeout):
        self._sock.settimeout(max(.005, timeout))
        try:
            data, (ip, _) = self._sock.recvfrom(64)
        except (TimeoutError, socket.timeout, OSError):
            return
        telemetry = decode_telemetry(data)
        with self._lock:
            if telemetry is not None and ip == self._ip:
                self._telemetry = telemetry
                self._last_rx = time.monotonic()

    def _run(self):
        next_send = 0.
        while not self._stop.is_set():
            now = time.monotonic()
            self._resolve(now)
            with self._lock:
                ip, wanted = self._ip, self._wanted
            if ip is not None and now >= next_send:
                self._send(ip, wanted)
                next_send = now + self._period
            if ip is None:
                self._stop.wait(.1)
            else:
                self._receive(next_send - time.monotonic())


def run_test(host, port=DEFAULT_PORT, cycles=3, seconds=2., echo=print):
    """Abre y cierra la pluma sin cámara, para probar cables y recorrido del servo."""
    link = BarrierLink(host, port)
    try:
        deadline = time.monotonic() + 5
        while not link.status().connected and time.monotonic() < deadline:
            time.sleep(.1)
        if not link.status().connected:
            raise RuntimeError(f'La pluma no responde ({link.status().describe()}). Revisar que la ESP32 esté en la '
                               'misma red y usar la IP que imprime el monitor serie con --host')
        for cycle in range(1, cycles + 1):
            for opened in (True, False):
                link.set_open(opened)
                echo(f"Ciclo {cycle}/{cycles}: {'abrir' if opened else 'cerrar'}")
                end = time.monotonic() + seconds
                while time.monotonic() < end:
                    time.sleep(.5)
                    echo('  ' + link.status().describe())
    finally:
        link.close()
