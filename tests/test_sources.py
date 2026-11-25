import base64
import http.client
import io
import cv2
import numpy as np
import pytest
from face_access.sources import MjpegCamera, build_request, iter_mjpeg, parse_camera, with_resolved_host


def jpeg(value):
    ok, data = cv2.imencode('.jpg', np.full((48, 64, 3), value, np.uint8))
    assert ok
    return data.tobytes()


def multipart(*frames):
    # Mismo formato que firmware/camara: CRLF, --frame, cabeceras, línea vacía y JPEG.
    body = b''
    for frame in frames:
        body += b'\r\n--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n' % len(frame) + frame
    return body


def test_parse_camera_accepts_index_or_url_only():
    assert parse_camera(0) == 0 and parse_camera('1') == 1
    assert parse_camera(' http://camara.local:81/stream ') == 'http://camara.local:81/stream'
    for bad in ('camara', 'rtsp://x', '', True):
        with pytest.raises(ValueError):
            parse_camera(bad)


def test_resolved_url_keeps_port_and_path():
    assert with_resolved_host('http://camara.local:81/stream', lambda h: '192.168.1.40') == 'http://192.168.1.40:81/stream'
    assert with_resolved_host('http://camara.local/stream', lambda h: None) is None


def test_iter_mjpeg_reads_parts_by_content_length():
    frames = [jpeg(10), jpeg(200)]
    stream = io.BytesIO(b'ruido previo\r\n' + multipart(*frames))
    assert list(iter_mjpeg(stream)) == frames


def test_iter_mjpeg_without_length_finds_jpeg_end_and_stops_on_truncation():
    # Otros servidores MJPEG omiten Content-Length: se corta en el marcador FF D9.
    frames = [jpeg(30), jpeg(220)]
    body = b''.join(b'--otro\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n' for frame in frames)
    assert list(iter_mjpeg(io.BytesIO(body))) == frames
    assert list(iter_mjpeg(io.BytesIO(b'--frame\r\nContent-Type: image/jpeg\r\n\r\nxx'))) == []
    data = multipart(jpeg(1))
    assert list(iter_mjpeg(io.BytesIO(data[:-5]))) == []
    with pytest.raises(ValueError):
        list(iter_mjpeg(io.BytesIO(b'--frame\r\nContent-Length: -4\r\n\r\n')))


def test_credentials_survive_resolution_and_become_basic_auth():
    url = with_resolved_host('http://yo:cl%40ve@telefono.local:8080/video', lambda h: '192.168.1.218')
    assert url == 'http://yo:cl%40ve@192.168.1.218:8080/video'
    request = build_request(url)
    assert request.full_url == 'http://192.168.1.218:8080/video'
    assert request.get_header('Authorization') == 'Basic ' + base64.b64encode(b'yo:cl@ve').decode()
    assert build_request('http://192.168.1.218:8080/video').get_header('Authorization') is None


class FakeResponse(io.BytesIO):
    def __init__(self, body, kind):
        super().__init__(body)
        self.headers = http.client.HTTPMessage()
        self.headers['Content-Type'] = kind


def test_web_page_instead_of_video_gives_route_hint():
    camera = MjpegCamera('http://192.168.1.218:8080', opener=lambda url: FakeResponse(b'<html></html>', 'text/html'),
                         first_frame_timeout=.3, retry=.05)
    try:
        assert not camera.isOpened()
        assert 'text/html' in camera.error and 'ruta del stream' in camera.error
    finally:
        camera.release()


def test_snapshot_url_is_polled_frame_by_frame():
    calls = []

    def opener(url):
        calls.append(url)
        return FakeResponse(jpeg(90 + len(calls) % 2 * 100), 'image/jpeg')

    camera = MjpegCamera('http://192.168.1.218:8080/shot.jpg', opener=opener, first_frame_timeout=2, retry=5)
    try:
        assert camera.isOpened() and camera.read()[0] and camera.read()[0]
        # retry=5 s: si la siguiente foto esperara el reintento, no habría llegado la segunda.
        assert len(calls) >= 2 and camera.connected
    finally:
        camera.release()


def test_ip_webcam_style_stream_with_headers():
    boundary = b'Ba4oTvQMY8ew04N8dcnM'
    body = b''.join(b'--' + boundary + b'\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n' % len(f) + f + b'\r\n'
                    for f in (jpeg(60), jpeg(160)))
    camera = MjpegCamera('http://192.168.1.218:8080/video',
                         opener=lambda url: FakeResponse(body, 'multipart/x-mixed-replace; boundary=' + boundary.decode()),
                         first_frame_timeout=2, retry=.05)
    try:
        ok, frame = camera.read()
        assert ok and frame.shape == (48, 64, 3)
    finally:
        camera.release()


def test_camera_returns_latest_frame_then_times_out_and_reconnects():
    attempts = []

    def opener(url):
        attempts.append(url)
        if len(attempts) == 1:
            raise OSError('HTTP Error 503: ya hay un cliente')
        return io.BytesIO(multipart(jpeg(50), jpeg(150)))

    camera = MjpegCamera('http://camara.local:81/stream', opener=opener, first_frame_timeout=2, frame_timeout=.3, retry=.05)
    try:
        assert camera.isOpened()
        ok, frame = camera.read()
        assert ok and frame.shape == (48, 64, 3)
        # Cada conexión de prueba termina enseguida, así que el hilo sigue reconectando.
        assert len(attempts) >= 2
    finally:
        camera.release()
    assert camera.read() == (False, None)


def test_camera_not_opened_keeps_error_for_the_user():
    def opener(url):
        raise OSError('No se encontró camara.local')

    camera = MjpegCamera('http://camara.local:81/stream', opener=opener, first_frame_timeout=.2, retry=.05)
    try:
        assert not camera.isOpened()
        assert 'camara.local' in camera.error
    finally:
        camera.release()
