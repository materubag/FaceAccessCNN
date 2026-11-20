import torch
from torch import nn

class FaceCNN(nn.Module):
    """CNN propia inicializada desde cero; nunca carga un backbone externo."""
    def __init__(self, channels=(32,64,128), kernel_size=3, padding=1, stride=1, hidden=64, dropout=.3, head="gap", input_size=96, num_classes=4):
        super().__init__()
        if head not in {"gap","flatten"}: raise ValueError("head debe ser gap o flatten")
        layers=[]; incoming=1; spatial=input_size
        for outgoing in channels:
            layers += [nn.Conv2d(incoming,outgoing,kernel_size,stride=stride,padding=padding), nn.ReLU(), nn.MaxPool2d(2)]
            incoming=outgoing
            spatial=((spatial+2*padding-kernel_size)//stride+1)//2
            if spatial<1: raise ValueError("Arquitectura incompatible con tamaño de entrada")
        self.features=nn.Sequential(*layers)
        self.pool=nn.AdaptiveAvgPool2d(1) if head=="gap" else nn.Identity()
        dense_input=incoming if head=="gap" else incoming*spatial*spatial
        self.classifier=nn.Sequential(nn.Flatten(),nn.Linear(dense_input,hidden),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden,num_classes))
    def forward(self,x): return self.classifier(self.pool(self.features(x)))

def describe(model, size=96):
    rows=[]; handles=[]
    def hook(name):
        def record(layer, inp, output):
            rows.append(dict(layer=name,type=type(layer).__name__,output=list(output.shape),
                parameters=sum(p.numel() for p in layer.parameters(recurse=False))))
        return record
    for name,layer in model.named_modules():
        if name and not list(layer.children()): handles.append(layer.register_forward_hook(hook(name)))
    training=model.training; model.eval()
    with torch.no_grad(): model(torch.zeros(1,1,size,size,device=next(model.parameters()).device))
    for h in handles: h.remove()
    model.train(training)
    for row in rows: print(row)
    print("Parámetros totales:",sum(p.numel() for p in model.parameters()))
    return rows
