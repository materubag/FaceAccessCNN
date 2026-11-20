from importlib.resources import files
from pathlib import Path


def detector_path():
    """El detector forma parte del paquete instalado, no del workspace académico."""
    return Path(str(files("face_access").joinpath("resources/yunet.onnx")))


def resolve(path, base):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (Path(base) / path).resolve()
