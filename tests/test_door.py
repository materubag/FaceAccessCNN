import pytest
from face_access.door import DoorState


def authorize(door,label=0,start=0):
    statuses=[door.step(label,1,start+i*.1) for i in range(5)]
    assert not any(s.opened for s in statuses[:-1]);assert statuses[-1].authorization_event
    return statuses[-1]


def test_open_scheduled_close_cooldown_and_new_authorization():
    door=DoorState();authorize(door)
    assert door.step(0,1,.5).state=='DOOR_OPEN'
    status=door.step(0,1,2.5);assert status.state=='COOLDOWN' and not status.opened
    for t in (2.6,2.7,2.8,3.):assert not door.step(0,1,t).opened
    assert door.step(0,1,5.4).authorization_event


@pytest.mark.parametrize('label,faces,quality,state',[(0,0,True,'NO_FACE'),(0,2,True,'MULTIPLE_FACES'),(3,1,True,'DENIED'),(0,1,False,'QUALITY_BLOCKED')])
def test_unsafe_sequence_always_closes_preserves_cooldown(label,faces,quality,state):
    door=DoorState();authorize(door);status=door.step(label,faces,.5,quality_ok=quality)
    assert status.state==state and not status.opened and status.valid_predictions==0
    assert status.cooldown_remaining>0


def test_change_identity_reset_pause_and_monotonic_clock():
    door=DoorState();door.step(0,1,0);assert door.step(1,1,.1).valid_predictions==1
    assert door.hold(.2).state=='PAUSED'
    with pytest.raises(ValueError):door.step(0,1,.1)


def test_new_authorized_class_and_unknown_last_index():
    door=DoorState(authorized_classes=4);authorize(door,label=3)
    paused=door.hold(.5);assert paused.state=='PAUSED' and not paused.opened
    assert door.step(4,1,.6).state=='DENIED'
