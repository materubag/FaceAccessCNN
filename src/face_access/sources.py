"""Fuentes de video: webcam local por índice o cámara de red por HTTP.

Cámaras de red: ESP32-CAM (firmware/camara) o un teléfono con IP Webcam. Aceptan un
stream MJPEG (multipart/x-mixed-replace) o una URL que devuelve una foto por petición.
Un hilo lee el video y conserva solo el último frame: si el reconocimiento va más lento
que la cámara se saltan frames en vez de acumular retraso.
"""
import base64
import socket
import ssl
import threading
import urllib.error
import urllib.request
from urllib.parse import unquote, urlsplit, urlunsplit
import cv2
import numpy as np

CONNECT_TIMEOUT_S = 3.
RETRY_S = 1.
FIRST_FRAME_TIMEOUT_S = 8.
FRAME_TIMEOUT_S = 2.
MDNS_TIMEOUT_MS = 1500
MAX_HEADER_LINES = 32
MAX_LINE_BYTES = 1024
MAX_FRAME_BYTES = 4 * 1024 * 1024


def is_stream_url(value):
    return isinstance(value, str) and value.strip().lower().startswith(('http://', 'https://'))


def parse_camera(value):
    """0 o '0' es una webcam local; 'http://...' es una ESP32-CAM. Lo demás es error."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    text = str(value).strip()
    if text.isdecimal():
        return int(text)
    if is_stream_url(text):
        return text
    raise ValueError(f'Cámara inválida: {value!r}; usar un índice (0, 1...) o una URL. {URL_HELP}')


def resolve_host(host):
    """IPv4 del host. Windows no siempre resuelve .local: se intenta mDNS con zeroconf si está."""
    try:
        return socket.getaddrinfo(host, None, socket.AF_INET, socket.SOCK_STREAM)[0][4][0]
    except (socket.gaierror, UnicodeError):
        pass
    if not host.endswith('.local'):
        return None
    try:
        from zeroconf import AddressResolver, Zeroconf
    except ImportError:
        return None
    zeroconf = Zeroconf()
    try:
        resolver = AddressResolver(host.rstrip('.') + '.')
        if resolver.request(zeroconf, MDNS_TIMEOUT_MS):
            addresses = resolver.parsed_addresses()
            return addresses[0] if addresses else None
    finally:
        zeroconf.close()
    return None


URL_HELP = ('ESP32-CAM: http://camara.local:81/stream | '
            'IP Webcam (Android): http://IP:8080/video (o http://IP:8080/shot.jpg)')


def with_resolved_host(url, resolver=resolve_host):
    """Cambia el nombre del host por su IP. Conserva usuario y clave si la URL los trae."""
    parts = urlsplit(url)
    if not parts.hostname:
        raise ValueError(f'URL sin host: {url}')
    ip = resolver(parts.hostname)
    if ip is None:
        return None
    netloc = f'{ip}:{parts.port}' if parts.port else ip
    if parts.username is not None:
        userinfo = parts.username + (f':{parts.password}' if parts.password is not None else '')
        netloc = f'{userinfo}@{netloc}'
    return urlunsplit(parts._replace(netloc=netloc))


def build_request(url):
    """Request sin credenciales en la URL: urllib no las usa solo, van como Basic auth."""
    parts = urlsplit(url)
    netloc = f'{parts.hostname}:{parts.port}' if parts.port else parts.hostname
    request = urllib.request.Request(urlunsplit(parts._replace(netloc=netloc)))
    if parts.username is not None:
        token = base64.b64encode(f'{unquote(parts.username)}:{unquote(parts.password or "")}'.encode()).decode()
        request.add_header('Authorization', 'Basic ' + token)
    return request


def _open_url(url):
    resolved = with_resolved_host(url)
    if resolved is None:
        raise OSError(f'No se encontró {urlsplit(url).hostname}; usar la IP de la cámara en la URL')
    try:
        return urllib.request.urlopen(build_request(resolved), timeout=CONNECT_TIMEOUT_S)
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise OSError('La cámara pide usuario y clave: usar http://usuario:clave@IP:puerto/ruta') from exc
        raise
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, ssl.SSLError):
            raise OSError('HTTPS no disponible (certificado autofirmado o el servidor solo habla http). Usar http://') from exc
        raise


class _Reader:
    """Buffer propio sobre el stream: deja leer líneas, N bytes o hasta el fin de un JPEG.

    Usa read1 para recibir lo que ya llegó sin esperar a llenar un bloque: si no, el final
    de cada frame quedaría retenido hasta que llegue el siguiente.
    """

    CHUNK = 16384

    def __init__(self, stream):
        self._stream = stream
        self._buffer = bytearray()

    def _fill(self):
        read = getattr(self._stream, 'read1', None) or self._stream.read
        data = read(self.CHUNK)
        if not data:
            return False
        self._buffer += data
        return True

    def readline(self, limit=MAX_LINE_BYTES):
        while True:
            end = self._buffer.find(b'\n', 0, limit)
            if end >= 0:
                return self._take(end + 1)
            if len(self._buffer) >= limit or not self._fill():
                return self._take(min(limit, len(self._buffer)))

    def read_exact(self, size):
        while len(self._buffer) < size:
            if not self._fill():
                return None
        return self._take(size)

    def read_jpeg(self):
        """Desde el inicio del JPEG (FF D8) hasta su marcador de fin (FF D9)."""
        searched = 0
        while True:
            start = self._buffer.find(b'\xff\xd8')
            if start >= 0:
                del self._buffer[:start]
                end = self._buffer.find(b'\xff\xd9', max(2, searched))
                if end >= 0:
                    return self._take(end + 2)
                searched = max(2, len(self._buffer) - 1)
            if len(self._buffer) > MAX_FRAME_BYTES:
                raise ValueError('Frame MJPEG demasiado grande o sin fin de JPEG')
            if not self._fill():
                return None

    def _take(self, size):
        data = bytes(self._buffer[:size])
        del self._buffer[:size]
        return data


def iter_mjpeg(stream):
    """Genera los JPEG de un multipart/x-mixed-replace.

    Usa el Content-Length de cada parte (ESP32-CAM, IP Webcam). Si una parte no lo trae,
    busca el fin del JPEG, como hacen otros servidores MJPEG.
    """
    reader = _Reader(stream)
    while True:
        line = reader.readline()
        if not line:
            return
        if not line.startswith(b'--'):
            continue
        length = None
        for _ in range(MAX_HEADER_LINES):
            header = reader.readline()
            if not header:
                return
            header = header.strip()
            if not header:
                break
            name, _, value = header.partition(b':')
            if name.strip().lower() == b'content-length':
                length = int(value)
        else:
            raise ValueError('Cabeceras MJPEG demasiado largas')
        if length is not None and not 0 < length <= MAX_FRAME_BYTES:
            raise ValueError('Parte MJPEG con Content-Length inválido')
        data = reader.read_exact(length) if length is not None else reader.read_jpeg()
        if data is None:
            return
        yield data


def content_type(response):
    """Tipo MIME de la respuesta, o None si el objeto no trae cabeceras (pruebas)."""
    headers = getattr(response, 'headers', None)
    return headers.get_content_type() if headers is not None else None


class MjpegCamera:
    """Misma interfaz que cv2.VideoCapture (isOpened, read, release) para un stream MJPEG.

    read() espera un frame nuevo hasta frame_timeout y devuelve (False, None) si no llega;
    mientras tanto el hilo sigue reconectando, así que una llamada posterior puede tener video.
    """

    def __init__(self, url, opener=_open_url, first_frame_timeout=FIRST_FRAME_TIMEOUT_S,
                 frame_timeout=FRAME_TIMEOUT_S, retry=RETRY_S):
        self.url = url
        self.error = None
        self._opener = opener
        self._frame_timeout = frame_timeout
        self._retry = retry
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._frame = None
        self._seq = 0
        self._read_seq = 0
        self._connected = False
        self._thread = threading.Thread(target=self._run, name='mjpeg', daemon=True)
        self._thread.start()
        with self._cond:
            self._cond.wait_for(lambda: self._seq > 0, timeout=first_frame_timeout)

    def isOpened(self):
        with self._cond:
            return self._seq > 0

    @property
    def connected(self):
        return self._connected

    def read(self):
        with self._cond:
            ready = self._cond.wait_for(lambda: self._seq > self._read_seq or self._stop.is_set(),
                                        timeout=self._frame_timeout)
            if not ready or self._stop.is_set():
                return False, None
            self._read_seq = self._seq
            return True, self._frame

    def release(self):
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        # La lectura bloqueada termina como mucho en CONNECT_TIMEOUT_S (timeout del socket).
        self._thread.join(timeout=CONNECT_TIMEOUT_S + 1)

    def _publish(self, frame):
        with self._cond:
            self._frame = frame
            self._seq += 1
            self._connected = True
            self.error = None
            self._cond.notify_all()

    def _decode(self, jpeg):
        frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if frame is not None:
            self._publish(frame)
        return frame is not None

    def _run(self):
        while not self._stop.is_set():
            wait = self._retry
            try:
                with self._opener(self.url) as response:
                    kind = content_type(response)
                    if kind == 'image/jpeg':
                        # Una foto por petición (p. ej. IP Webcam /shot.jpg): se pide la siguiente enseguida.
                        if self._decode(response.read(MAX_FRAME_BYTES + 1)):
                            wait = 0
                        else:
                            self.error = 'La URL devolvió una imagen ilegible'
                    elif kind is not None and not kind.startswith('multipart/'):
                        raise ValueError(f'La URL devolvió {kind} (una página), no video: falta la ruta del stream')
                    else:
                        for jpeg in iter_mjpeg(response):
                            if self._stop.is_set():
                                return
                            self._decode(jpeg)
                        self.error = 'La cámara cerró el stream'
            except (OSError, ValueError) as exc:
                # Incluye 503 (otro cliente ya ve el stream), timeouts y cortes de WiFi.
                self.error = str(exc)
            if wait:
                self._connected = False
            self._stop.wait(wait)
