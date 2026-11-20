"""Una única demo para modelos importados o propios; espejo exclusivamente visual."""
from contextlib import nullcontext
import json
from pathlib import Path
import time
import cv2
import numpy as np
from .detection import ProfilePipeline
from .door import DoorState
from .display import render_demo
from .inference import Predictor,classification


def run_demo(checkpoint,calibration,settings,preview_only=False,events=None):
    predictor=None if preview_only else Predictor(checkpoint,calibration,settings['device'])
    if predictor is not None and predictor.checkpoint['smoke_only']:raise ValueError('Un checkpoint smoke no puede autorizar aperturas')
    if predictor is not None and predictor.calibration is None:raise ValueError('Primero evaluar/calibrar con val')
    labels=predictor.labels if predictor else ['Autorizado','Desconocido']
    pipeline=ProfilePipeline('yunet_v1',deduplicate=False)
    door=DoorState(consistent_frames=settings['consistent_frames'],open_seconds=settings['open_seconds'],cooldown_seconds=settings['cooldown_seconds'],authorized_classes=len(labels)-1)
    if door.required<1 or door.duration<=0 or door.cooldown<door.duration:raise ValueError('Temporizadores/confirmaciones inválidos')
    model_name='SOLO DETECCION: sin identidad' if predictor is None else ('EXPERIMENTAL ' if predictor.checkpoint['experimental'] else '')+predictor.checkpoint['model_id'].upper()
    threshold=predictor.calibration['threshold'] if predictor else 1.;margin=predictor.calibration['margin'] if predictor else .15
    stream=nullcontext(None)
    if events:
        path=Path(events)
        if path.suffix!='.jsonl':raise ValueError('Eventos requieren .jsonl')
        path.parent.mkdir(parents=True,exist_ok=True);stream=path.open('x',encoding='utf-8')
    camera=None;previous=None
    print(model_name,'| Sin detección de vida; puerta simulada. D diagnóstico; Q salir.')
    try:
        with stream as logger:
            camera=cv2.VideoCapture(settings['camera'])
            if not camera.isOpened():raise RuntimeError('Webcam no disponible; probar --camera 1 y permisos de Windows')
            while True:
                ok,frame=camera.read()
                if not ok:break
                detected=pipeline.process(frame);count=detected['face_count'];crop=None
                boxes=[r['detection'].box for r in detected['results'] if np.asarray(r['detection'].box).shape==(4,) and np.isfinite(r['detection'].box).all()]
                landmarks=[r['detection'].landmarks for r in detected['results'] if r['detection'].landmarks is not None]
                detector_score=detected['results'][0]['detection'].score if count==1 else None
                if detector_score is not None and not np.isfinite(detector_score):detector_score=None
                quality=dict(status=detected['status'],ok=False,reasons=sorted({reason for r in detected['results'] for reason in r['reasons']}),
                    detector_score=detector_score,threshold=threshold,minimum_margin=margin)
                if count==1 and detected['status']=='candidate':crop=detected['results'][0]['crop']
                quality['ok']=crop is not None
                probability=predictor.predict(crop) if predictor is not None and crop is not None else None
                details=classification(probability,threshold,margin,len(labels));now=time.monotonic()
                status=door.hold(now) if preview_only else door.step(details['label'],count,now,quality_ok=quality['ok'],rejection_reason=details['reason'])
                if logger and (status.transition or previous!=(details['predicted'],details['reason'],quality['status'])):
                    logger.write(json.dumps(dict(model=model_name,state=status.state,reason=status.reason,predicted=details['predicted'],score=details['score'],margin=details['margin'],
                        faces=count,quality=quality['status'],opened=status.opened,authorization=status.authorization_event,monotonic_time=now,
                        checkpoint_sha256=getattr(predictor,'checkpoint_sha256',None),calibration_sha256=getattr(predictor,'calibration_sha256',None)),allow_nan=False)+'\n');logger.flush()
                previous=(details['predicted'],details['reason'],quality['status'])
                view=render_demo(frame,boxes,status,details,quality,count,settings['debug'],mirror=settings['mirror'],landmarks=landmarks,model_banner=model_name,labels=labels)
                cv2.imshow('FaceAccessCNN | Q salir',view);key=cv2.waitKey(1)&255
                if key in (ord('q'),ord('Q')):break
                if key in (ord('d'),ord('D')):settings['debug']=not settings['debug']
                if key in (ord('m'),ord('M')):settings['mirror']=not settings['mirror']
                try:
                    if cv2.getWindowProperty('FaceAccessCNN | Q salir',cv2.WND_PROP_VISIBLE)<1:break
                except cv2.error:break
    finally:
        if camera is not None:camera.release()
        cv2.destroyAllWindows()
