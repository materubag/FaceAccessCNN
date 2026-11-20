from pathlib import Path
import cv2
import numpy as np

def read_image(path):
    # Compatible con rutas Unicode de Windows.
    raw=np.fromfile(str(path), dtype=np.uint8)
    if raw.size==0:raise ValueError(f"Imagen vacía: {path}")
    try:arr=cv2.imdecode(raw, cv2.IMREAD_COLOR)
    except cv2.error as exc:raise ValueError(f"Imagen ilegible: {path}") from exc
    if arr is None: raise ValueError(f"Imagen ilegible: {path}")
    return arr

def base_array(image, filter_name="median", size=96, detector=None):
    bgr=read_image(image) if isinstance(image,(str,Path)) else image
    if bgr is None or bgr.size==0: raise ValueError("Imagen vacía")
    if detector is not None: bgr=detector.single(bgr)
    gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY) if bgr.ndim==3 else bgr.copy()
    gray=cv2.resize(gray,(size,size),interpolation=cv2.INTER_AREA)
    if filter_name=="median": gray=cv2.medianBlur(gray,3)
    elif filter_name=="gaussian": gray=cv2.GaussianBlur(gray,(3,3),0)
    elif filter_name!="none": raise ValueError(f"Filtro desconocido: {filter_name}")
    return np.ascontiguousarray(gray.astype(np.float32)/255.)

def tensor_image(image, filter_name="median", size=96, detector=None, augment=False, generator=None):
    import torch
    arr=base_array(image,filter_name,size,detector)
    if augment:
        # Sólo entrenamiento: espejo y variación moderada de iluminación.
        if torch.rand(1, generator=generator).item()<.5: arr=arr[:,::-1].copy()
        arr=np.clip(arr*(.85+.3*torch.rand(1, generator=generator).item()),0,1)
    return torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0)

def blur_score(bgr):
    gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(cv2.resize(gray,(96,96)),cv2.CV_64F).var())

def difference_hash(bgr):
    gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
    tiny=cv2.resize(gray,(9,8))
    return int.from_bytes(np.packbits(tiny[:,1:]>tiny[:,:-1]).tobytes(),"big")
