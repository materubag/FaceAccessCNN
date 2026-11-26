"""Una única demo para modelos importados o propios; espejo exclusivamente visual."""
from contextlib import nullcontext
import json
from pathlib import Path
import time
import cv2
import numpy as np
from .barrier import DEFAULT_PORT,BarrierLink
from .detection import ProfilePipeline
from .door import DoorState
from .display import render_demo
from .inference import Predictor,classification
from .sources import URL_HELP,MjpegCamera,is_stream_url,parse_camera


def run_demo(checkpoint,calibration,settings,preview_only=False,events=None):
    supervised=bool(settings.get('supervised_actuation',False))
    if preview_only and supervised:raise ValueError('La vista previa no admite accionamiento supervisado')
    predictor=None if preview_only else Predictor(checkpoint,calibration,settings['device'])
    if predictor is not None and predictor.checkpoint['smoke_only']:raise ValueError('Un checkpoint smoke no puede autorizar aperturas')
    if predictor is not None and predictor.calibration is None:raise ValueError('Primero evaluar/calibrar con val')
    labels=predictor.labels if predictor else ['Autorizado','Desconocido']
    pipeline=ProfilePipeline('yunet_v1',deduplicate=False)
    temporal=dict(consistent_frames=settings['consistent_frames'],open_seconds=settings['open_seconds'],cooldown_seconds=settings['cooldown_seconds'],
        recognition_seconds=settings.get('recognition_seconds',1.5))
    if predictor:temporal.update(predictor.checkpoint.get('temporal_policy',{}))
    model_approval_passed=bool(predictor is not None and getattr(predictor,'authorization_enabled',True))
    authorization_enabled=bool(predictor is not None and (model_approval_passed or supervised))
    door=DoorState(**temporal,authorized_classes=len(labels)-1,authorization_enabled=authorization_enabled)
    if door.required<1 or door.duration<=0 or door.cooldown<door.duration:raise ValueError('Temporizadores/confirmaciones inválidos')
    model_name='SOLO DETECCION: sin identidad' if predictor is None else ('EXPERIMENTAL ' if predictor.checkpoint['experimental'] else '')+predictor.checkpoint['model_id'].upper()
    if predictor is not None and not authorization_enabled:model_name+=' | SOLO DIAGNOSTICO - APERTURA BLOQUEADA'
    if supervised:
        model_name+=' | ACCIONAMIENTO SUPERVISADO'
        print('PRUEBA SUPERVISADA: se habilita el servo pese a la falta de aprobación del modelo. Puede abrir para una identidad incorrecta. No cambian score, margen, calidad ni confirmaciones.')
        if events is None:
            import uuid
            events=Path('runs/demo_events')/('supervised_'+uuid.uuid4().hex+'.jsonl')
        print('Registro de la prueba:',events)
    threshold=predictor.calibration['threshold'] if predictor else 1.;margin=predictor.calibration['margin'] if predictor else .15
    stream=nullcontext(None)
    if events:
        path=Path(events)
        if path.suffix!='.jsonl':raise ValueError('Eventos requieren .jsonl')
        path.parent.mkdir(parents=True,exist_ok=True);stream=path.open('x',encoding='utf-8')
    camera=None;previous=None;barrier=None;last_shape=(480,640,3)
    source=parse_camera(settings['camera']);streaming=is_stream_url(source)
    actuator='pluma '+settings['barrier_host'] if settings.get('barrier_host') and not preview_only and authorization_enabled else 'puerta simulada'
    if settings.get('barrier_host') and not authorization_enabled:print('Modo diagnóstico: no se conecta ni envía comandos a la pluma.')
    print(model_name,f'| Sin detección de vida; {actuator}. D diagnóstico; Q salir.')
    try:
        with stream as logger:
            backend=settings.get('camera_backend','auto')
            if streaming:
                print('Conectando con',source)
                camera=MjpegCamera(source)
            elif backend=='auto':
                camera=cv2.VideoCapture(source)
            elif backend=='dshow':
                camera=cv2.VideoCapture(source,cv2.CAP_DSHOW)
            elif backend=='msmf':
                camera=cv2.VideoCapture(source,cv2.CAP_MSMF)
            else:
                raise ValueError('Backend de cámara inválido; usar auto, dshow o msmf')
            if not camera.isOpened():
                if streaming:raise RuntimeError(f'No llega video de {source}: {camera.error or "sin respuesta"}. Revisar que la cámara esté encendida y en la misma red. Rutas: {URL_HELP}')
                raise RuntimeError('Webcam no disponible; probar --camera 1 y permisos de Windows')
            # Después de abrir la cámara: así la pluma no recibe comandos si no hay video.
            if settings.get('barrier_host') and not preview_only and authorization_enabled:barrier=BarrierLink(settings['barrier_host'],settings.get('barrier_port',DEFAULT_PORT))
            while True:
                ok,frame=camera.read()
                if not ok:
                    if not streaming:break
                    # Sin video de la ESP32-CAM: cerrar, reiniciar la secuencia y esperar a que vuelva.
                    now=time.monotonic();status=door.hold(now) if preview_only else door.step(None,0,now)
                    if barrier:barrier.set_open(False)
                    view=no_video_view(last_shape,camera.error,barrier.status().describe() if barrier else None)
                    if not show(view,settings):break
                    continue
                last_shape=frame.shape
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
                if barrier:barrier.set_open(status.opened)
                if logger and (status.transition or previous!=(details['predicted'],details['reason'],quality['status'])):
                    logger.write(json.dumps(dict(model=model_name,state=status.state,reason=status.reason,predicted=details['predicted'],score=details['score'],margin=details['margin'],
                        faces=count,quality=quality['status'],opened=status.opened,authorization=status.authorization_event,monotonic_time=now,
                        supervised_actuation=supervised,model_approval_passed=model_approval_passed,
                        checkpoint_sha256=getattr(predictor,'checkpoint_sha256',None),calibration_sha256=getattr(predictor,'calibration_sha256',None)),allow_nan=False)+'\n');logger.flush()
                previous=(details['predicted'],details['reason'],quality['status'])
                view=render_demo(frame,boxes,status,details,quality,count,settings['debug'],mirror=settings['mirror'],landmarks=landmarks,model_banner=model_name,labels=labels)
                if barrier:draw_footer(view,barrier.status().describe(),barrier.status().connected)
                if not show(view,settings):break
    finally:
        if barrier is not None:barrier.close()
        if camera is not None:camera.release()
        cv2.destroyAllWindows()


WINDOW='FaceAccessCNN | Q salir'


def show(view,settings):
    """Muestra la vista y atiende teclas. False si hay que salir."""
    cv2.imshow(WINDOW,view);key=cv2.waitKey(1)&255
    if key in (ord('q'),ord('Q')):return False
    if key in (ord('d'),ord('D')):settings['debug']=not settings['debug']
    if key in (ord('m'),ord('M')):settings['mirror']=not settings['mirror']
    try:return cv2.getWindowProperty(WINDOW,cv2.WND_PROP_VISIBLE)>=1
    except cv2.error:return False


def draw_footer(view,text,ok):
    """Estado de la pluma justo encima de la línea de ayuda de render_demo."""
    cv2.putText(view,text,(10,view.shape[0]-32),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,200,0) if ok else (0,140,255),1,cv2.LINE_AA)


def no_video_view(shape,error,barrier_text):
    view=np.full(shape,25,dtype=np.uint8)
    lines=['SIN VIDEO DE LA CAMARA DE RED - reconectando','Pluma cerrada. Q: salir']
    if error:lines.append(str(error)[:70])
    for i,line in enumerate(lines):cv2.putText(view,line,(20,40+i*30),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,140,255),1,cv2.LINE_AA)
    if barrier_text:draw_footer(view,barrier_text,False)
    return view
