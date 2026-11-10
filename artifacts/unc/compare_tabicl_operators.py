import json
import numpy as np
a=np.load('artifacts/unc/operator_samples_local.npz')
b=np.load(r'C:\Users\LDY\Downloads\operator_samples_remote.npz')
r={}
for k in a.files:
    x=a[k].astype(float); y=b[k].astype(float); d=abs(x-y)
    r[k]={'max':float(d.max()),'mean':float(d.mean()),'different':int((d!=0).sum()),'n':int(d.size)}
print(json.dumps(r,indent=2))
