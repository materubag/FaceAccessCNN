"""Detección/ROI únicamente. No reconoce identidades ni carga clasificadores externos."""
from dataclasses import dataclass
from pathlib import Path
import math
import time
import cv2
import numpy as np
from .common import digest
from .config import detector_profile
from .paths import detector_path
from .preprocessing import blur_score,difference_hash

@dataclass
class Detection:
    box: tuple
    score: float | None = None
    landmarks: np.ndarray | None = None

class Detector:
    def __init__(self,profile):
        self.cfg=profile;self.name=profile["detector"]
        if self.name=="yunet":
            path=detector_path()
            if digest(path)!=profile["model_sha256"]:raise ValueError("SHA256 del detector incorrecto")
            if not hasattr(cv2,"FaceDetectorYN"):raise RuntimeError("OpenCV no soporta FaceDetectorYN")
            self.model=cv2.FaceDetectorYN.create(str(path),"",(320,320),profile["candidate_score"],
                profile["nms_threshold"],profile["top_k"],cv2.dnn.DNN_BACKEND_OPENCV,cv2.dnn.DNN_TARGET_CPU)
        else:raise ValueError("Detector no soportado")
    def detect(self,frame):
        factor=min(1.,self.cfg["max_width"]/frame.shape[1])
        image=cv2.resize(frame,None,fx=factor,fy=factor) if factor<1 else frame
        self.model.setInputSize((image.shape[1],image.shape[0]));_,faces=self.model.detect(image)
        if faces is None:return []
        return [Detection(tuple(float(v/factor) for v in face[:4]),float(face[14]),
            np.asarray(face[4:14],dtype=float).reshape(5,2)/factor) for face in faces]

def load_profile(name="yunet_v1"):
    if name!="yunet_v1":raise ValueError("Solo YuNet es compatible")
    return detector_profile()


def assess_geometry(shape,detection,cfg):
    """Landmarks dudosos sólo señalan revisión; nunca se rechaza por perfil."""
    height,width=shape[:2];reasons=[]
    box=np.asarray(detection.box,dtype=float)
    if len(box)!=4 or not np.isfinite(box).all() or min(box[2:])<=0:
        return dict(status="rejected",reasons=["invalid_bbox"],visible_fraction=0.,landmarks_consistent=False)
    x,y,w,h=box
    visible=max(0,min(width,x+w)-max(0,x))*max(0,min(height,y+h)-max(0,y))/(w*h)
    if visible==0:return dict(status="rejected",reasons=["bbox_outside_image"],visible_fraction=0.,landmarks_consistent=False)
    if min(w,h)<cfg["min_face_pixels"]:reasons.append("face_too_small_review")
    if not cfg["aspect_range"][0]<=w/h<=cfg["aspect_range"][1]:reasons.append("aspect_ratio_review")
    if visible<cfg["min_visible_bbox_fraction"]:reasons.append("truncated_face_review")
    if detection.score is None:reasons.append("confidence_unavailable_review")
    elif not np.isfinite(detection.score) or detection.score<cfg["accept_score"]:reasons.append("low_confidence_review")
    lm=detection.landmarks;consistent=False
    if lm is None:reasons.append("landmarks_unavailable_review")
    else:
        lm=np.asarray(lm,dtype=float)
        if lm.shape!=(5,2):reasons.append("landmark_shape_review")
        else:
            finite=np.isfinite(lm).all(axis=1)
            # YuNet no proporciona visibilidad/confianza por landmark.
            inside=finite&(lm[:,0]>=max(0,x-.15*w))&(lm[:,0]<=min(width,x+1.15*w))&(lm[:,1]>=max(0,y-.15*h))&(lm[:,1]<=min(height,y+1.15*h))
            if not inside.all():reasons.append("partial_or_occluded_landmarks_review")
            if finite.all():
                eyes=lm[:2].mean(axis=0);mouth=lm[3:].mean(axis=0);nose=lm[2]
                eye_span=np.linalg.norm(lm[0]-lm[1])/w
                plausible=.04<=eye_span<=1.2 and eyes[1]<=nose[1]+.25*h and mouth[1]>=nose[1]-.15*h and mouth[1]-eyes[1]>=.05*h
                if not plausible:reasons.append("landmark_geometry_review")
                consistent=bool(plausible and inside.all())
            else:reasons.append("landmarks_nonfinite_review")
    return dict(status="review" if reasons else "candidate",reasons=reasons,visible_fraction=float(visible),landmarks_consistent=consistent)

def square_crop(frame,box,cfg,return_original=False):
    """Cuadrado con margen proporcional; relleno gris explícito, sin deformar el ROI."""
    values=np.asarray(box,dtype=float)
    if values.shape!=(4,) or not np.isfinite(values).all() or min(values[2:])<=0:raise ValueError('Bounding box inválido')
    bx,by,bw,bh=values;fh,fw=frame.shape[:2]
    if min(fw,bx+bw)<=max(0,bx) or min(fh,by+bh)<=max(0,by):raise ValueError('ROI fuera de imagen')
    x,y,w,h=map(float,box);side=int(math.ceil(max(w,h)*(1+2*cfg["margin"])))
    if side<1:raise ValueError("ROI vacío")
    height,width=frame.shape[:2];left=int(math.floor(x+w/2-side/2));top=int(math.floor(y+h/2-side/2))
    # Desplazar un cuadrado que sí cabe conserva contexto sin padding innecesario.
    if side<=width:left=min(max(0,left),width-side)
    if side<=height:top=min(max(0,top),height-side)
    x0,y0=max(0,left),max(0,top);x1,y1=min(width,left+side),min(height,top+side)
    if x1<=x0 or y1<=y0:raise ValueError("ROI fuera de imagen")
    # No reservar memoria según una caja arbitrariamente grande de un detector defectuoso.
    if side>4*max(width,height):raise ValueError("ROI excesivo")
    square=np.full((side,side,3),cfg["padding_value"],dtype=np.uint8)
    square[y0-top:y1-top,x0-left:x1-left]=frame[y0:y1,x0:x1]
    padding=1-(x1-x0)*(y1-y0)/(side*side)
    crop=cv2.resize(square,(cfg["output_size"],cfg["output_size"]),interpolation=cv2.INTER_AREA)
    metadata=dict(square_box=[left,top,side,side],padding_fraction=float(padding),size=cfg["output_size"],margin=cfg["margin"])
    return (crop,metadata,square) if return_original else (crop,metadata)

class ProfilePipeline:
    def __init__(self,profile,detector=None,deduplicate=True,include_original_roi=False):
        self.cfg=load_profile(profile) if isinstance(profile,str) else profile
        if self.cfg.get("align_eyes"):raise ValueError("Alineación no validada")
        self.detector=detector or Detector(self.cfg);self.previous=[];self.deduplicate=deduplicate
        self.include_original_roi=include_original_roi
    def process(self,frame):
        started=time.perf_counter();detections=self.detector.detect(frame);detection_seconds=time.perf_counter()-started;results=[]
        for detection in detections:
            assessment=assess_geometry(frame.shape,detection,self.cfg)
            if len(detections)!=1:
                assessment["reasons"].append("multiple_faces_review")
                if assessment["status"]!="rejected":assessment["status"]="review"
            crop=None;crop_metadata={};blur=None;dhash=None;original_roi=None
            if assessment["status"]!="rejected":
                try:
                    if self.include_original_roi:
                        crop,crop_metadata,original_roi=square_crop(frame,detection.box,self.cfg,return_original=True)
                    else:crop,crop_metadata=square_crop(frame,detection.box,self.cfg)
                except ValueError as exc:assessment.update(status="rejected",reasons=assessment["reasons"]+[str(exc)])
            if crop is not None:
                if crop_metadata["padding_fraction"]>self.cfg["max_crop_padding_fraction"]:
                    assessment["reasons"].append("excessive_padding_review");assessment["status"]="review"
                blur=blur_score(crop);dhash=difference_hash(crop)
                if blur<self.cfg["blur_threshold"]:
                    assessment["reasons"].append("blurry_review");assessment["status"]="review"
                if assessment["status"]=="candidate" and self.deduplicate:
                    if any((dhash^old).bit_count()<=self.cfg["duplicate_distance"] for old in self.previous):
                        assessment["status"]="duplicate";assessment["reasons"].append("near_duplicate")
                    else:self.previous.append(dhash)
            results.append(dict(detection=detection,crop=crop,crop_metadata=crop_metadata,blur=blur,dhash=dhash,**assessment))
            if self.include_original_roi:results[-1]["original_roi"]=original_roi
        return dict(detection_seconds=detection_seconds,processing_seconds=time.perf_counter()-started,face_count=len(detections),status="no_face" if not detections else ("multiple_faces_review" if len(detections)>1 else results[0]["status"]),results=results)

def annotate(frame,result):
    image=frame.copy()
    for r in result["results"]:
        d=r["detection"]
        if not np.isfinite(d.box).all():continue
        x,y,w,h=map(lambda v:int(round(v)),d.box)
        color=(0,180,0) if r["status"]=="candidate" else (0,180,255)
        cv2.rectangle(image,(x,y),(x+w,y+h),color,2)
        if d.landmarks is not None:
            for point in d.landmarks:
                if np.isfinite(point).all():cv2.circle(image,tuple(int(round(v)) for v in point),3,(255,80,30),-1)
        caption=r["status"]+(f" {d.score:.3f}" if d.score is not None else "")
        cv2.putText(image,caption,(max(0,x),max(18,y-5)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
    return image
