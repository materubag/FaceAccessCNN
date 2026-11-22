"""Adquisición consentida y revisión explícita; nunca aprueba por score."""
import json
from pathlib import Path
import re
import time
import uuid
import cv2
import numpy as np
from .common import canonical, digest, read_json, save_json
from .config import contract_hash
from .detection import ProfilePipeline, annotate
from .preprocessing import read_image


def identifier(value):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value): raise ValueError("Usa identificadores anónimos de1-64 letras/números/guiones")
    return value


def capture(person, session, source="0", data_root="data/captures", unknown=False,
            split=None, consent=False, independent=False, fps=3., limit=180, preview=True):
    if not consent: raise ValueError("Captura requiere --consent y permisos legítimos sobre las imágenes")
    if not independent: raise ValueError('Captura de dataset requiere --independent-session; para solo ver cámara usa demo --preview-only')
    if not 2 <= fps <= 5 or limit < 1: raise ValueError("FPS entre2 y5; límite positivo")
    person=identifier(person); session=identifier(session)
    if unknown and split not in {"train","val","test"}: raise ValueError("Desconocidos requieren --split train/val/test")
    if not unknown and split is not None: raise ValueError("El split autorizado se fija por sesión al preparar")
    root=Path(data_root).resolve();branch=root/"unknown"/split/person/session if unknown else root/"authorized"/person/session
    directory=branch/uuid.uuid4().hex[:12];directory.mkdir(parents=True,exist_ok=False)
    pipeline=ProfilePipeline("yunet_v1",deduplicate=True);inventory=[];processed=0;cap=None
    source_path=Path(source);is_camera=str(source).isdecimal();stills=sorted(p for p in source_path.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png'}) if source_path.is_dir() else None
    source_id="camera:"+uuid.uuid4().hex if is_camera else "source:"+digest(source_path) if stills is None else None
    def store(frame,index,seconds,origin):
        nonlocal processed
        result=pipeline.process(frame);processed+=1
        if result['face_count']==1 and result['status']=='candidate':
            crop=result['results'][0]['crop'];path=directory/f"frame_{index:06d}.jpg"
            ok,encoded=cv2.imencode('.jpg',crop,[cv2.IMWRITE_JPEG_QUALITY,95])
            if not ok: raise RuntimeError('Error codificando recorte')
            encoded.tofile(str(path));sha=digest(path)
            row=dict(sample_id=canonical([origin,index,sha])[:24],path=path.relative_to(root).as_posix(),sha256=sha,
                identity=person,session=session,kind='unknown' if unknown else 'authorized',split=split,
                source_id=origin,source_frame=index,source_seconds=seconds,automatic_status='candidate',
                score=float(result['results'][0]['detection'].score),contract_sha256=contract_hash())
            inventory.append(row)
            # Persist accepted candidates incrementally; an incomplete run remains unusable.
            save_json(directory/'inventory.json',inventory)
        if preview:
            cv2.imshow('Adquisicion: candidatos pendientes de revision | Q salir',annotate(frame,result))
            if cv2.waitKey(1)&255 in (ord('q'),ord('Q')):return False
            try:return cv2.getWindowProperty('Adquisicion: candidatos pendientes de revision | Q salir',cv2.WND_PROP_VISIBLE)>=1
            except cv2.error:return False
        return True
    complete=False
    try:
        if stills is not None:
            for index,path in enumerate(stills):
                if processed>=limit or not store(read_image(path),index,0.,'image:'+digest(path)):break
        else:
            cap=cv2.VideoCapture(int(source) if is_camera else str(source_path))
            if not cap.isOpened(): raise RuntimeError('No se pudo abrir la cámara/video; comprobar --source y permisos')
            source_fps=cap.get(cv2.CAP_PROP_FPS)
            if not is_camera and (not np.isfinite(source_fps) or source_fps<=0):raise ValueError('Video sin FPS válidos')
            next_sample=0.;index=-1;started=time.monotonic()
            while processed<limit:
                ok,frame=cap.read()
                if not ok:break
                index+=1;seconds=time.monotonic()-started if is_camera else index/source_fps
                if seconds<next_sample:continue
                next_sample=seconds+1/fps
                if not store(frame,index,seconds,source_id):break
        complete=True
    finally:
        if cap is not None:cap.release()
        if preview:cv2.destroyAllWindows()
        save_json(directory/'run.json',dict(complete=complete,consent_confirmed=consent,independence_confirmed=independent,
            identity=person,session=session,kind='unknown' if unknown else 'authorized',split=split,
            sampled_frames=processed,candidates=len(inventory),contract_sha256=contract_hash()))
    print(f"Candidatos pendientes: {len(inventory)} / muestreados: {processed}. Revisar antes de preparar.")
    return directory


def load_reviews(root):
    decisions={};file=Path(root)/'reviews.jsonl'
    if file.exists():
        for line in file.read_text(encoding='utf-8').splitlines():
            event=json.loads(line);event_id=event.pop('event_id')
            if canonical(event)!=event_id or event['status'] not in {'accepted','rejected','pending'}:raise ValueError('Evento de revisión alterado')
            event['event_id']=event_id;decisions[event['sample_id']]=event
    return decisions


def catalog(root):
    root=Path(root);records=[]
    for inventory in sorted(root.glob('**/inventory.json')):
        run=read_json(inventory.parent/'run.json')
        if not run['complete']:raise ValueError('Captura incompleta: '+str(inventory.parent))
        if not run['consent_confirmed'] or not run['independence_confirmed']:raise ValueError('Falta consentimiento/confirmación de sesión independiente: '+str(inventory.parent))
        if run['contract_sha256']!=contract_hash():raise ValueError('Perfil de adquisición distinto')
        for row in read_json(inventory):
            if any(row[key]!=run[key] for key in ('identity','session','kind','split','contract_sha256')):raise ValueError('Inventario y fuente no coinciden')
            path=(root/row['path']).resolve();path.relative_to(root.resolve())
            if digest(path)!=row['sha256'] or row['automatic_status']!='candidate':raise ValueError('Recorte modificado o no elegible')
            records.append(row)
    return records


def decide(root,row,status):
    root=Path(root)
    if status not in {'accepted','rejected','pending'}:raise ValueError('Estado de revisión inválido')
    if digest(root/row['path'])!=row['sha256']:raise ValueError('Imagen modificada')
    event=dict(sample_id=row['sample_id'],image_sha256=row['sha256'],status=status,reviewer='human',event_nonce=uuid.uuid4().hex)
    event['event_id']=canonical(event)
    import os
    with (root/'reviews.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(event,ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno())


def review(data_root,person=None):
    root=Path(data_root).resolve();records=[r for r in catalog(root) if person is None or r['identity']==person]
    if not records:raise ValueError('No hay candidatos para revisar')
    index=0
    try:
        while True:
            row=records[index];latest=load_reviews(root).get(row['sample_id'],{}).get('status','pending')
            image=cv2.resize(read_image(root/row['path']),(576,576),interpolation=cv2.INTER_NEAREST)
            image=cv2.copyMakeBorder(image,75,0,0,0,cv2.BORDER_CONSTANT,value=(25,25,25))
            cv2.putText(image,f"{index+1}/{len(records)} {row['identity']} {row['session']} {latest}",(10,25),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            cv2.putText(image,'A aceptar | R rechazar | P pendiente | J/L navegar | Q salir',(10,55),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1)
            cv2.imshow('Revision manual de candidatos',image);key=cv2.waitKey(30)&255
            if key in (ord('q'),ord('Q')):break
            try:
                if cv2.getWindowProperty('Revision manual de candidatos',cv2.WND_PROP_VISIBLE)<1:break
            except cv2.error:break
            if key in (ord('a'),ord('r'),ord('p')):
                decide(root,row,{ord('a'):'accepted',ord('r'):'rejected',ord('p'):'pending'}[key]);index=min(index+1,len(records)-1)
            elif key in (ord('l'),83):index=min(index+1,len(records)-1)
            elif key in (ord('j'),81):index=max(0,index-1)
    finally:cv2.destroyAllWindows()
