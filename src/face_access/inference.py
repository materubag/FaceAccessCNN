"""Carga vinculada a calibración; decisiones no dependen de la identidad verdadera."""
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from .common import digest, canonical, read_json
from .config import PREPROCESSING, detector_profile, input_contract, contract_hash
from .model import FaceCNN
from .preprocessing import tensor_image, read_image

FORMAT = "face-access-cnn/1"
MODELS = {
    "r02": dict(source="6f506781acfe84b55a5e34a91182742baed6431b99ce5eb6b43c67c3918a8228", calibration="0ba6e864c0e150c94123f450a3c753740740e163b6161ec4a0965b183358a99c", threshold=.9197969436645508, weights="830e9986f5e60550c54748cce9b82d8c17ffc0b5fb224a133110c4e2ddb0d848"),
    "r03b": dict(source="7a4c31380cf33508d62a47c89b8642b989af0a87ef4f1084a3fa61a00c9b943a", calibration="8feb07771448045623ff146ba4f48a14ba6196ef7c0c0748d702b630dfb44b00", threshold=.9813899397850037, weights="f10d384c8bd58f5d7012ce976f160e68e2493c5cfc1268e00889bda0e5b46be0"),
}


def weights_digest(weights):
    import hashlib
    h = hashlib.sha256()
    for name, tensor in sorted(weights.items()):
        value = tensor.detach().cpu().contiguous()
        h.update(json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


def preprocess(crop):
    image = read_image(crop) if isinstance(crop, (str, Path)) else crop
    if image is None or image.shape != (96, 96, 3) or image.dtype != np.uint8:
        raise ValueError("Se requiere ROI BGR uint8 de96×96, no fotograma completo")
    return tensor_image(image, "median", 96)


def decisions(probabilities, threshold, margin=.15):
    probability = np.asarray(probabilities)
    if probability.ndim != 2 or probability.shape[1] < 2 or not np.isfinite(probability).all() or (probability<0).any() or (probability>1).any() or not np.allclose(probability.sum(1),1,atol=1e-5):
        raise ValueError("Probabilidades inválidas")
    predicted = probability.argmax(1)
    ordered = np.sort(probability, axis=1)
    unknown = probability.shape[1] - 1
    accepted = (predicted != unknown) & (ordered[:, -1] > threshold) & ((ordered[:, -1] - ordered[:, -2]) > margin)
    return np.where(accepted, predicted, unknown)


def classification(probability, threshold, margin, classes):
    if probability is None:
        return dict(predicted=None, label=classes-1, score=None, margin=None, reason="not_classified_this_frame")
    probability = np.asarray(probability)
    if probability.shape != (classes,) or not np.isfinite(probability).all() or (probability<0).any() or (probability>1).any() or not np.isclose(probability.sum(),1,atol=1e-5):
        return dict(predicted=None, label=classes-1, score=None, margin=None, reason="invalid_probabilities")
    predicted = int(probability.argmax()); ordered = np.sort(probability)
    score = float(probability[predicted]); gap = float(ordered[-1]-ordered[-2])
    accepted = int(decisions(probability[None], threshold, margin)[0])
    reason = "predicted_unknown" if predicted == classes-1 else "score_below_threshold" if not score > threshold else "margin_below_minimum" if not gap > margin else "accepted_classifier"
    return dict(predicted=predicted, label=accepted, score=score, margin=gap, reason=reason)


def import_model(checkpoint, calibration, directory, name, safety_attestation=None):
    """Importación LOCAL de archivos de confianza; nunca descarga pesos sensibles."""
    if name not in MODELS:
        return import_local_experiment(checkpoint,calibration,directory,name,safety_attestation)
    expected = MODELS[name]; checkpoint = Path(checkpoint); calibration = Path(calibration)
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    original_sha = ck.get("source_checkpoint_sha256", digest(checkpoint))
    if original_sha != expected["source"] or digest(calibration) != expected["calibration"]:
        raise ValueError("No corresponde al modelo/calibración seleccionados")
    cal = read_json(calibration)
    if cal["checkpoint_sha256"] != original_sha or cal["manifest_sha256"] != ck["manifest_sha256"] or cal["calibrated_split"] != "val":
        raise ValueError("Calibración ajena al checkpoint")
    if cal["threshold"] != expected["threshold"] or cal["margin"] != .15: raise ValueError("Calibración congelada distinta")
    old_contract = ck.get("input_contract", {})
    if canonical(old_contract) != ck.get("input_contract_sha256") or old_contract.get("preprocessing") != PREPROCESSING:
        raise ValueError("Contrato de entrada no compatible")
    old_detector = old_contract.get("detection_profile", {})
    if any(old_detector.get(key) != value for key, value in detector_profile().items()):
        raise ValueError("Perfil YuNet no compatible")
    if weights_digest(ck["weights"]) != expected["weights"]: raise ValueError("Tensores distintos al modelo seleccionado")
    if len(ck["labels"]) != 4 or ck["config"]["model"].get("head") != "flatten": raise ValueError("Arquitectura/etiquetas incompatibles")
    folder = Path(directory); target = folder / (name+".pt"); target_cal = folder / (name+"_calibration.json")
    if target.exists() or target_cal.exists(): raise FileExistsError("Conservar modelo local existente")
    folder.mkdir(parents=True, exist_ok=True)
    artifact = dict(format=FORMAT, model_config=dict(ck["config"]["model"], num_classes=4), labels=ck["labels"],
        input_contract=input_contract(), contract_sha256=contract_hash(), weights=ck["weights"], weights_sha256=expected["weights"],
        source_checkpoint_sha256=original_sha, manifest_sha256=ck["manifest_sha256"], smoke_only=bool(ck["smoke_only"]),
        experimental=name == "r03b", model_id=name)
    torch.save(artifact, target); shutil.copyfile(calibration, target_cal)
    return target, target_cal


def import_local_experiment(checkpoint,calibration,directory,name,safety_attestation=None):
    """Conversión local de confianza; sin datos privados ni política permisiva por defecto."""
    import re
    if not re.fullmatch(r'[a-z][a-z0-9_-]{0,39}',name):raise ValueError('Nombre local inválido')
    checkpoint=Path(checkpoint);calibration=Path(calibration)
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True);cal=read_json(calibration)
    source=digest(checkpoint)
    if (cal.get('checkpoint_sha256')!=source or cal.get('manifest_sha256')!=ck.get('manifest_sha256') or
            cal.get('calibrated_split')!='val' or ck.get('smoke_only') is not False or ck.get('mode')!='real'):
        raise ValueError('Experimento/calibración incompatibles o modelo smoke/provisional')
    old=ck.get('input_contract',{})
    if canonical(old)!=ck.get('input_contract_sha256') or old.get('preprocessing')!=PREPROCESSING:
        raise ValueError('Contrato original incompatible')
    if any(old.get('detection_profile',{}).get(k)!=v for k,v in detector_profile().items()):
        raise ValueError('Perfil YuNet incompatible')
    if ck.get('labels')!=['Persona 1','Persona 2','Persona 3','Desconocido'] or ck['config']['model'].get('head')!='flatten':
        raise ValueError('Etiquetas/arquitectura incompatibles')
    # Missing, stale or failed evidence keeps every experimental import diagnostic.
    gate=read_json(safety_attestation) if safety_attestation else {}
    allowed=bool(gate.get('allowed') is True and gate.get('checkpoint_sha256')==source and
        gate.get('calibration_policy_sha256')==canonical(cal) and
        gate.get('criteria_sha256')==canonical(ck['config'].get('safety_gate_criteria')) and
        gate.get('checks') and all(v is True for v in gate['checks'].values()))
    folder=Path(directory);target=folder/(name+'.pt');target_cal=folder/(name+'_calibration.json')
    if target.exists() or target_cal.exists():raise FileExistsError('Conservar modelo local existente')
    folder.mkdir(parents=True,exist_ok=True)
    temporal=dict(ck['config']['decision'])
    temporal={k:temporal[k] for k in ('consistent_frames','open_seconds','cooldown_seconds')}
    temporal['recognition_seconds']=cal.get('temporal_policy',{}).get('recognition_seconds',1.5)
    artifact=dict(format=FORMAT,model_config=dict(ck['config']['model'],num_classes=4),labels=ck['labels'],
        input_contract=input_contract(),contract_sha256=contract_hash(),weights=ck['weights'],weights_sha256=weights_digest(ck['weights']),
        source_checkpoint_sha256=source,manifest_sha256=ck['manifest_sha256'],smoke_only=False,experimental=True,model_id=name,
        frozen_calibration_sha256=digest(calibration),requires_safety_gate=True,temporal_policy=temporal,
        safety_policy=dict(allowed=allowed,checks=gate.get('checks',{}),
            source_attestation_sha256=digest(safety_attestation) if safety_attestation else None))
    torch.save(artifact,target);shutil.copyfile(calibration,target_cal)
    return target,target_cal


class Predictor:
    def __init__(self, checkpoint, calibration=None, device="auto"):
        self.path = Path(checkpoint)
        if not self.path.exists(): raise FileNotFoundError("Modelo local ausente. Usa 'models import' con autorización o entrena tu propio modelo; consulta models/README.md")
        self.checkpoint_sha256=digest(self.path);self.calibration_sha256=None
        self.checkpoint = torch.load(self.path, map_location="cpu", weights_only=True)
        ck = self.checkpoint
        if ck.get("format") != FORMAT or ck.get("contract_sha256") != contract_hash() or ck.get("input_contract") != input_contract():
            raise ValueError("Formato o contrato incompatible; usa models import para artefactos antiguos")
        if weights_digest(ck["weights"]) != ck["weights_sha256"]: raise ValueError("Pesos modificados")
        self.labels = ck["labels"]
        if len(self.labels) < 2 or len(set(self.labels)) != len(self.labels) or ck["model_config"]["num_classes"] != len(self.labels):
            raise ValueError("Mapeo de clases inválido")
        self.device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device)
        self.model = FaceCNN(**ck["model_config"]).to(self.device)
        self.model.load_state_dict(ck["weights"]); self.model.eval(); self.calibration = None
        self.authorization_enabled=not ck.get('requires_safety_gate',False)
        if calibration is not None:
            cal = read_json(calibration)
            allowed={digest(self.path)}
            if ck.get('source_checkpoint_sha256'):allowed.add(ck['source_checkpoint_sha256'])
            if cal.get("checkpoint_sha256") not in allowed or cal.get("manifest_sha256") != ck["manifest_sha256"] or cal.get("calibrated_split") != "val":
                raise ValueError("La calibración no pertenece a este modelo y validación")
            if not 0 <= cal["threshold"] <= 1.000001 or not 0 <= cal["margin"] <= 1: raise ValueError("Umbral/margen inválidos")
            if ck.get("model_id") in MODELS:
                spec = MODELS[ck["model_id"]]
                if digest(calibration) != spec["calibration"]: raise ValueError("Calibración congelada modificada")
            if ck.get('frozen_calibration_sha256') and digest(calibration)!=ck['frozen_calibration_sha256']:
                raise ValueError('Calibración congelada del experimento modificada')
            self.calibration = cal
            self.calibration_sha256=digest(calibration)
            if ck.get('requires_safety_gate'):
                policy=ck.get('safety_policy',{})
                self.authorization_enabled=bool(policy.get('allowed') is True and policy.get('checks') and all(v is True for v in policy['checks'].values()))

    @torch.inference_mode()
    def predict(self, crop):
        x = preprocess(crop).unsqueeze(0).to(self.device)
        return self.model(x).softmax(1)[0].cpu().numpy()
