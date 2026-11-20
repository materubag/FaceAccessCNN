import time
from dataclasses import dataclass,replace
import numpy as np

@dataclass(frozen=True)
class DoorStatus:
    state: str
    opened: bool
    authorization_event: bool
    reason: str
    identity: int | None
    valid_predictions: int
    required_predictions: int
    open_remaining: float
    cooldown_remaining: float
    transition: bool
    last_transition: dict
    monotonic_time: float

class DoorState:
    """Cooldown desde la ultima autorizacion; no equivale a rechazo de identidad.

    update conserva la API (opened, authorization_event). step expone diagnostico.
    Las causas inseguras siempre cierran, reinician el contador y conservan cooldown.
    """
    def __init__(self,consistent_frames=5,cooldown_seconds=5,open_seconds=2, authorized_classes=3):
        self.authorized_classes=authorized_classes; self.required=consistent_frames; self.cooldown=cooldown_seconds; self.duration=open_seconds
        self.identity=None; self.count=0; self.last_open=-float("inf"); self.until=0
        self.state="START";self.last_transition={};self.last_time=None
    def update(self,label,face_count,now=None):
        status=self.step(label,face_count,now)
        return status.opened,status.authorization_event

    def hold(self,now=None):
        """Pausa explicita de ensayos: cierra y reinicia secuencia, conserva cooldown."""
        previous=self.state;previous_transition=dict(self.last_transition)
        status=self.step(self.authorized_classes,1,now=now,rejection_reason="operator_paused")
        transition=previous!="PAUSED"
        self.state="PAUSED"
        if transition:self.last_transition=dict(from_state=previous,to_state="PAUSED",reason="operator_paused",at=status.monotonic_time)
        else:self.last_transition=previous_transition
        return replace(status,state="PAUSED",reason="operator_paused",transition=transition,last_transition=dict(self.last_transition))

    def step(self,label,face_count,now=None,quality_ok=True,rejection_reason=None):
        now=time.monotonic() if now is None else now
        if not np.isfinite(now) or (self.last_time is not None and now<self.last_time):raise ValueError("Reloj monotono invalido")
        self.last_time=now;event=False;old_state=self.state;old_identity=self.identity
        expired=old_state in {"AUTHORIZED","DOOR_OPEN"} and self.until>0 and now>=self.until
        if face_count==0:
            state="NO_FACE";reason="no_face"
        elif face_count!=1:
            state="MULTIPLE_FACES";reason="multiple_faces"
        elif not quality_ok:
            state="QUALITY_BLOCKED";reason="quality_not_candidate"
        elif label not in range(self.authorized_classes):
            state="DENIED";reason=rejection_reason or "unknown_or_classifier_rejected"
        else:state=None
        if state is not None:
            self.identity=None; self.count=0; self.until=0
        else:
            changed=label!=self.identity
            if changed:self.identity=label;self.count=1;self.until=0
            else:self.count=min(self.required,self.count+1)
            if now<self.until:
                state="DOOR_OPEN";reason="open_timer_active"
            elif now-self.last_open<self.cooldown:
                state="COOLDOWN";reason="open_timer_elapsed" if expired else "identity_changed" if changed and old_identity is not None else "cooldown_active"
            elif self.count<self.required:
                state="VERIFYING";reason="identity_changed" if changed and old_identity is not None else "waiting_consistent_predictions"
            else:
                state="AUTHORIZED";reason="consecutive_predictions_confirmed"
                self.last_open=now;self.until=now+self.duration;event=True;self.count=0
        self.state=state
        transition=state!=old_state or self.identity!=old_identity or event
        if transition:self.last_transition=dict(from_state=old_state,to_state=state,reason=reason,at=now)
        return DoorStatus(state=state,opened=now<self.until,authorization_event=event,reason=reason,identity=self.identity,
            valid_predictions=self.required if event else self.count,required_predictions=self.required,open_remaining=max(0.,self.until-now),
            cooldown_remaining=max(0.,self.last_open+self.cooldown-now),transition=transition,
            last_transition=dict(self.last_transition),monotonic_time=now)
