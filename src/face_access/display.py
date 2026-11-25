import textwrap
import cv2
import numpy as np

STATE_TEXT={"NO_FACE":"BLOQUEADO - SIN ROSTRO","MULTIPLE_FACES":"MULTIPLES ROSTROS",
    'DIAGNOSTIC_ONLY':'SOLO DIAGNOSTICO - PUERTA BLOQUEADA',
    "QUALITY_BLOCKED":"BLOQUEADO POR CALIDAD","DENIED":"DENEGADO","VERIFYING":"VERIFICANDO",
    "AUTHORIZED":"AUTORIZADO - PUERTA ABIERTA","DOOR_OPEN":"PUERTA ABIERTA","COOLDOWN":"PUERTA CERRADA / COOLDOWN","PAUSED":"PRUEBA PAUSADA"}

def display_box(box,width,mirror=True):
    x,y,w,h=box
    return (width-x-w,y,w,h) if mirror else (x,y,w,h)

def display_point(point,width,mirror=True):
    x,y=point
    return (width-1-x,y) if mirror else (x,y)

def render_demo(frame,boxes,status,classification,quality,face_count,debug=False,mirror=True,landmarks=None,trial_banner=None,model_banner=None,labels=None):
    labels = labels or ["Persona 1","Persona 2","Persona 3","Desconocido"]
    # Solo presentacion: el fotograma de entrada no se modifica ni se devuelve al detector.
    image=cv2.flip(frame,1) if mirror else frame.copy();state=status.state
    color=(0,180,0) if status.opened else (200,160,30) if state=="COOLDOWN" else (0,180,255) if state in {"VERIFYING","QUALITY_BLOCKED"} else (0,0,220)
    for box in boxes:
        x,y,w,h=map(int,display_box(box,frame.shape[1],mirror))
        if w>0 and h>0:cv2.rectangle(image,(x,y),(x+w-1,y+h-1),color,2)
    if debug:
        for group in landmarks or []:
            for point in group:
                if np.isfinite(point).all():cv2.circle(image,tuple(int(round(v)) for v in display_point(point,frame.shape[1],mirror)),2,(255,150,40),-1)
    predicted=classification["predicted"]
    identity=labels[predicted] if predicted is not None else "SIN CLASIFICAR (fotograma actual)"
    score="--" if classification["score"] is None else f"{classification['score']:.3f}"
    gap="--" if classification["margin"] is None else f"{classification['margin']:.3f}"
    # HUD compacto superpuesto; texto/puerta se dibujan despues del espejo y son legibles.
    compact={"NO_FACE":"SIN ROSTRO","AUTHORIZED":"PUERTA ABIERTA","COOLDOWN":"COOLDOWN - PUERTA CERRADA"}
    access="PRUEBA PAUSADA" if trial_banner and trial_banner.startswith("PAUSADO") else compact.get(state,STATE_TEXT[state])
    name="DESCONOCIDO" if predicted==len(labels)-1 else identity
    hud=[f"{name} | score {score} | rostros {face_count}",access]
    if model_banner:hud.append(model_banner)
    if trial_banner:hud.append(trial_banner)
    max_chars=max(25,(image.shape[1]-24)//8);hud=[part for line in hud for part in textwrap.wrap(line,max_chars)]
    panel_height=min(image.shape[0],len(hud)*23+12)
    overlay=image[:panel_height].copy();overlay[:]=20
    image[:panel_height]=cv2.addWeighted(image[:panel_height],.2,overlay,.8,0)
    for i,line in enumerate(hud):cv2.putText(image,line,(10,23+i*23),cv2.FONT_HERSHEY_SIMPLEX,.52,color,1,cv2.LINE_AA)
    cv2.putText(image,"SIMULACION | D: debug | M: espejo | Q: salir",(10,image.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,.42,(230,230,230),1,cv2.LINE_AA)
    right=image.shape[1]-12;left=max(0,right-42);top=min(panel_height+8,max(0,image.shape[0]-90))
    cv2.rectangle(image,(left,top),(right,top+64),color,-1 if status.opened else 2)
    if not debug:return image
    lines=[f"Identificacion: {identity} | score={score}",f"Deteccion: {quality['status']} | rostros={face_count}",
        "Acceso: "+STATE_TEXT[state],"SIMULACION - SIN DETECCION DE VIDA - q: salir"]
    if state=="COOLDOWN":lines.append(f"Identidad conservada; espera {status.cooldown_remaining:.2f}s. No es rechazo.")
    elif state=="VERIFYING":lines.append(f"Confirmando secuencia: {status.valid_predictions}/{status.required_predictions}")
    if debug:
        tr=status.last_transition
        lines += [f"Estado={state} | causa={status.reason}",
            f"Ultima: {tr.get('from_state')} -> {tr.get('to_state')} / {tr.get('reason')}",
            f"Validas={status.valid_predictions}/{status.required_predictions} | apertura={status.open_remaining:.2f}s | cooldown={status.cooldown_remaining:.2f}s",
            f"CNN: {classification['reason']} | margen={gap} | min_score={quality['threshold']:.6f} | min_margen={quality['minimum_margin']:.2f}",
            "Filtros: "+("; ".join(quality["reasons"]) or "sin alertas"),
            "Score detector: "+("--" if quality.get("detector_score") is None else f"{quality['detector_score']:.3f}")]
    width=max(24,(frame.shape[1]-28)//8);wrapped=[]
    for line in lines:wrapped.extend(textwrap.wrap(line,width=width) or [""])
    panel=np.full((len(wrapped)*23+18,frame.shape[1],3),24,dtype=np.uint8)
    for i,line in enumerate(wrapped):cv2.putText(panel,line,(12,24+i*23),cv2.FONT_HERSHEY_SIMPLEX,.51,color if i<3 else (235,235,235),1,cv2.LINE_AA)
    return np.vstack((panel,image))
