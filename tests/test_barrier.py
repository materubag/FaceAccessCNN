import socket
import time
import pytest
from face_access.barrier import (FLAG_AT_TARGET, FLAG_FAILSAFE, FLAG_OPENING, MAGIC_COMMAND, MAGIC_TELEMETRY, TELEMETRY,
                                 BarrierLink, BarrierStatus, crc8, decode_telemetry, encode_command)


def telemetry(last_seq, flags, rssi=-55, version=1):
    body = TELEMETRY.pack(MAGIC_TELEMETRY, version, last_seq, rssi, flags, 0)[:-1]
    return body + bytes([crc8(body)])


def test_crc8_matches_firmware_polynomial():
    # Valor de comprobación estándar de CRC-8 (poly 0x07, init 0x00), igual que protocol.cpp.
    assert crc8(b'123456789') == 0xF4


def test_command_layout_matches_firmware_struct():
    packet = encode_command(0x1234, True)
    assert len(packet) == 6
    assert packet[:5] == bytes([MAGIC_COMMAND, 1, 0x34, 0x12, 1])
    assert packet[5] == crc8(packet[:5])
    assert encode_command(65536, False)[2:5] == bytes([0, 0, 0])


def test_decode_telemetry_rejects_corrupt_packets():
    good = telemetry(7, FLAG_OPENING | FLAG_AT_TARGET)
    decoded = decode_telemetry(good)
    assert decoded.last_seq == 7 and decoded.rssi == -55 and decoded.flags == FLAG_OPENING | FLAG_AT_TARGET
    assert decode_telemetry(good[:-1] + bytes([good[-1] ^ 1])) is None
    assert decode_telemetry(good[:-1]) is None
    assert decode_telemetry(telemetry(7, 0, version=2)) is None


def test_status_text_covers_each_state():
    assert 'buscando' in BarrierStatus().describe()
    assert 'sin respuesta' in BarrierStatus(ip='10.0.0.2').describe()
    assert 'failsafe' in BarrierStatus(connected=True, failsafe=True, ip='10.0.0.2').describe()
    assert 'abriendo' in BarrierStatus(connected=True, failsafe=False, opening=True, rssi=-50, ip='10.0.0.2').describe()


@pytest.fixture
def fake_barrier():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('127.0.0.1', 0))
    sock.settimeout(2)
    yield sock
    sock.close()


def wait_for(condition, timeout=2.):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(.01)
    return False


def test_link_sends_wanted_state_receives_telemetry_and_closes_on_exit(fake_barrier):
    link = BarrierLink('127.0.0.1', fake_barrier.getsockname()[1], period=.02)
    try:
        data, peer = fake_barrier.recvfrom(64)
        assert data[0] == MAGIC_COMMAND and data[4] == 0

        link.set_open(True)
        assert wait_for(lambda: fake_barrier.recvfrom(64)[0][4] == 1)
        _, peer = fake_barrier.recvfrom(64)
        fake_barrier.sendto(telemetry(1, FLAG_OPENING), peer)
        assert wait_for(lambda: link.status().connected)
        status = link.status()
        assert status.opening and not status.failsafe and not status.at_target and status.rssi == -55
    finally:
        link.close()

    # Tras close(), los últimos paquetes piden cerrar y la secuencia sigue subiendo.
    last = None
    fake_barrier.settimeout(.3)
    try:
        while True:
            last = fake_barrier.recvfrom(64)[0]
    except (TimeoutError, socket.timeout):
        pass
    assert last is not None and last[4] == 0


def test_link_ignores_telemetry_from_other_hosts_and_failsafe_flag(fake_barrier):
    link = BarrierLink('127.0.0.1', fake_barrier.getsockname()[1], period=.02)
    try:
        _, peer = fake_barrier.recvfrom(64)
        fake_barrier.sendto(b'basura', peer)
        fake_barrier.sendto(telemetry(1, FLAG_FAILSAFE), peer)
        assert wait_for(lambda: link.status().connected)
        assert link.status().failsafe
    finally:
        link.close()


def test_link_resolves_hostname_with_injected_resolver(fake_barrier):
    calls = []

    def resolver(host):
        calls.append(host)
        return '127.0.0.1'

    link = BarrierLink('pluma.local', fake_barrier.getsockname()[1], resolver=resolver, period=.02)
    try:
        assert fake_barrier.recvfrom(64)[0][0] == MAGIC_COMMAND
        assert calls == ['pluma.local']
    finally:
        link.close()
