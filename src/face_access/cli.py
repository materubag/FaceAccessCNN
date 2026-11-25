import argparse
import json
from pathlib import Path
from .config import load_config
from .paths import resolve


def parser():
    p=argparse.ArgumentParser(prog='face-access',description='CNN propia y YuNet; puerta únicamente simulada')
    sub=p.add_subparsers(dest='command',required=True)
    demo=sub.add_parser('demo',help='Webcam con r03b/r02 o checkpoint propio')
    demo.add_argument('--config',type=Path);demo.add_argument('--model',choices=['r03b','r02']);demo.add_argument('--model-dir',type=Path)
    demo.add_argument('--checkpoint',type=Path);demo.add_argument('--calibration',type=Path);demo.add_argument('--camera',type=int)
    demo.add_argument('--camera-backend',choices=['auto','dshow','msmf'])
    demo.add_argument('--device',choices=['auto','cpu','cuda']);demo.add_argument('--no-mirror',action='store_true');demo.add_argument('--debug',action='store_true')
    demo.add_argument('--preview-only',action='store_true');demo.add_argument('--check',action='store_true',help='Cargar/validar sin abrir cámara');demo.add_argument('--events',type=Path)
    cap=sub.add_parser('capture',help='Capturar candidatos con consentimiento; no aprueba imágenes')
    cap.add_argument('--person',required=True);cap.add_argument('--session',required=True);cap.add_argument('--source',default='0')
    cap.add_argument('--data-root',default='data/captures');cap.add_argument('--unknown',action='store_true');cap.add_argument('--split',choices=['train','val','test'])
    cap.add_argument('--consent',action='store_true');cap.add_argument('--independent-session',action='store_true');cap.add_argument('--fps',type=float,default=3)
    cap.add_argument('--limit',type=int,default=180);cap.add_argument('--no-preview',action='store_true')
    review=sub.add_parser('review',help='Revisar y guardar decisiones humanas append-only')
    review.add_argument('--data-root',default='data/captures');review.add_argument('--person')
    prep=sub.add_parser('prepare',help='Generar/auditar manifiesto train/val/test');prep.add_argument('--config',type=Path,default=Path('configs/train.yaml'))
    train=sub.add_parser('train',help='Entrenar desde cero; nunca reutiliza pesos')
    train.add_argument('--config',type=Path,default=Path('configs/train.yaml'));train.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');train.add_argument('--smoke',action='store_true')
    evaluate=sub.add_parser('evaluate',help='Validación/calibración o test una vez tras congelación')
    evaluate.add_argument('--checkpoint',type=Path,required=True);evaluate.add_argument('--manifest',type=Path,required=True);evaluate.add_argument('--output',type=Path,required=True)
    evaluate.add_argument('--split',choices=['val','test'],default='val');evaluate.add_argument('--calibration',type=Path);evaluate.add_argument('--confirm-frozen',action='store_true')
    evaluate.add_argument('--target-far',type=float);evaluate.add_argument('--margin',type=float);evaluate.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    models=sub.add_parser('models',help='Gestionar pesos sensibles solo de forma local');actions=models.add_subparsers(dest='action',required=True)
    listing=actions.add_parser('list');listing.add_argument('--model-dir',type=Path,default=Path('models'))
    imp=actions.add_parser('import',help='Importar artefactos de confianza con permiso de uso LOCAL')
    imp.add_argument('--name',choices=['r02','r03b'],required=True);imp.add_argument('--checkpoint',type=Path,required=True);imp.add_argument('--calibration',type=Path,required=True);imp.add_argument('--model-dir',type=Path,default=Path('models'))
    sub.add_parser('self-test',help='YuNet/CNN/tensor sintéticos CPU; sin cámara ni datos privados')
    return p


def main(argv=None):
    p=parser();args=p.parse_args(argv)
    try:
        if args.command=='demo':
            from .demo import run_demo
            settings=dict(model='r03b',model_dir='models',camera=0,camera_backend='auto',device='auto',mirror=True,debug=False,
                consistent_frames=5,recognition_seconds=1.5,open_seconds=2.,cooldown_seconds=5.)
            default=Path('configs/demo.yaml')
            path=args.config or (default if default.exists() else None)
            if path:
                settings.update(load_config(path))
                settings['model_dir']=resolve(settings['model_dir'],path.resolve().parent.parent)
            for field in ('model','model_dir','camera','camera_backend','device'):
                if getattr(args,field) is not None:settings[field]=getattr(args,field)
            if args.no_mirror:settings['mirror']=False
            if args.debug:settings['debug']=True
            if args.model is not None and args.checkpoint is not None:raise ValueError('Elegir --model o --checkpoint')
            checkpoint=args.checkpoint or Path(settings['model_dir'])/(settings['model']+'.pt')
            calibration=args.calibration or (checkpoint.parent/(settings['model']+'_calibration.json') if args.checkpoint is None else None)
            if args.check:
                from .inference import Predictor
                if args.preview_only:raise ValueError('Para prueba de detector sin modelo: self-test')
                predictor=Predictor(checkpoint,calibration,settings['device'])
                if predictor.calibration is None or predictor.checkpoint['smoke_only']:raise ValueError('Checkpoint sin calibración usable o smoke')
                print(json.dumps(dict(ok=True,model=predictor.checkpoint['model_id'],labels=predictor.labels,threshold=predictor.calibration['threshold'],margin=predictor.calibration['margin'],camera_opened=False),ensure_ascii=False))
            else:run_demo(checkpoint,calibration,settings,args.preview_only,args.events)
        elif args.command=='capture':
            from .acquisition import capture
            print(capture(args.person,args.session,args.source,args.data_root,args.unknown,args.split,args.consent,args.independent_session,args.fps,args.limit,not args.no_preview))
        elif args.command=='review':
            from .acquisition import review
            review(args.data_root,args.person)
        elif args.command in {'prepare','train'}:
            cfg=load_config(args.config);base=args.config.resolve().parent.parent
            if args.command=='prepare':
                from .dataset import prepare
                print(prepare(cfg,base))
            else:
                from .training import train
                print(train(cfg,resolve(cfg['manifest'],base),resolve(cfg['output'],base),args.smoke,args.device))
        elif args.command=='evaluate':
            from .evaluation import evaluate
            result=evaluate(args.checkpoint,args.manifest,args.output,args.split,args.calibration,args.target_far,args.margin,args.confirm_frozen,args.device)
            print(json.dumps({k:result[k] for k in ('split','accuracy','macro_f1','far','frr')},indent=2))
        elif args.command=='models':
            from .inference import MODELS,import_model
            if args.action=='list':
                print(json.dumps({name:dict(threshold=item['threshold'],weights_distributed=False,
                    local_available=(args.model_dir/(name+'.pt')).exists() and (args.model_dir/(name+'_calibration.json')).exists()) for name,item in MODELS.items()},indent=2))
            else:print(import_model(args.checkpoint,args.calibration,args.model_dir,args.name))
        elif args.command=='self-test':
            import numpy as np
            import torch
            from .detection import Detector,load_profile
            from .inference import preprocess
            from .model import FaceCNN
            detector=Detector(load_profile());assert detector.detect(np.zeros((240,320,3),np.uint8))==[]
            tensor=preprocess(np.random.default_rng(42).integers(0,256,(96,96,3),dtype=np.uint8))
            with torch.inference_mode():output=FaceCNN(head='flatten').eval()(tensor[None])
            print(json.dumps(dict(ok=True,device='cpu',tensor_shape=list(tensor.shape),output_shape=list(output.shape),camera_opened=False,identity_model_used=False)))
    except (ValueError,FileNotFoundError,FileExistsError,RuntimeError,KeyError) as exc:p.error(str(exc))

if __name__=='__main__':main()
