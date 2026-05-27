from astropy.io import fits
import numpy as np

files=[
("F390W","data/multiband/F390W.fits",390),
("F435W","data/multiband/F435W.fits",435),
("F555W","data/multiband/F555W.fits",555),
("F814W","data/multiband/F814W.fits",814),
]

Cs=[]
lam=[]

for name,f,l in files:

    d=fits.getdata(f)
    d=np.nan_to_num(d)

    y,x=np.indices(d.shape)

    xc=d.shape[1]/2
    yc=d.shape[0]/2

    r=np.sqrt((x-xc)**2+(y-yc)**2)

    # annulus approximate arc region
    m=(r>200)&(r<500)

    v=d[m]
    v=v[np.isfinite(v)]

    C=np.std(v)/np.mean(np.abs(v))

    print(name,l,"C =",C)

    lam.append(l)
    Cs.append(C)

lam=np.array(lam,dtype=float)
Cs=np.array(Cs)

coef=np.polyfit(np.log(lam),np.log(Cs),1)

print("\nC(lambda)=A lambda^alpha")
print("alpha =",coef[0])
