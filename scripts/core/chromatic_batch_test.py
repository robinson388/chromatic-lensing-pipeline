from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

ra=246.94354
dec=-0.89933
pos=SkyCoord(ra*u.deg,dec*u.deg)

files=[
("F390W","data/j1627_multiband/F390W.fits",390,2.0),
("F435W","data/j1627_multiband/F435W.fits",435,2.5),
("F555W","data/j1627_multiband/F555W.fits",555,3.0),
("F814W","data/j1627_multiband/F814W.fits",814,3.5),
]

res=[]

for band,fn,lam,sig in files:

    h=fits.open(fn)
    data=np.asarray(h["SCI"].data,dtype=float)
    data=np.nan_to_num(data)

    w=WCS(h["SCI"].header)

    cut=Cutout2D(data,pos,size=(12*u.arcsec,12*u.arcsec),wcs=w)
    img=cut.data

    p1,p99=np.percentile(img,[1,99])
    img=np.clip(img,p1,p99)
    img=img-img.min()
    if img.max()>0:
        img=img/img.max()

    # modelo suave de galaxia deflectora
    model=median_filter(img,size=31)

    # residuo positivo del arco
    resid=img-model
    resid=resid-np.percentile(resid,5)
    resid=np.clip(resid,0,None)
    if resid.max()>0:
        resid=resid/resid.max()

    sm=gaussian_filter(resid,sigma=sig)

    y,x=np.indices(sm.shape)
    cx=sm.shape[1]/2
    cy=sm.shape[0]/2
    r=np.sqrt((x-cx)**2+(y-cy)**2)

    ann=(r>14)&(r<45)
    vals=sm[ann]

    if vals.size == 0 or np.all(vals==0):
        continue

    thr=np.percentile(vals,85)
    mask=ann&(sm>=thr)

    arc=sm[mask]

    Ccontrast=np.std(arc)/(np.mean(np.abs(arc))+1e-12)

    gy,gx=np.gradient(sm)
    rx=x-cx
    ry=y-cy

    cs=[]
    wt=[]

    ys,xs=np.where(mask)

    for yy,xx in zip(ys,xs):
        rn=np.hypot(rx[yy,xx],ry[yy,xx])
        gn=np.hypot(gx[yy,xx],gy[yy,xx])
        if rn>0 and gn>0:
            c=(rx[yy,xx]*(-gx[yy,xx])+ry[yy,xx]*(-gy[yy,xx]))/(rn*gn)
            cs.append(c)
            wt.append(sm[yy,xx])

    cs=np.array(cs)
    wt=np.array(wt)

    A=np.average(cs,weights=wt)
    Cdir=abs(A)
    Astd=np.sqrt(np.average((cs-A)**2,weights=wt))

    res.append((band,lam,Ccontrast,Cdir,Astd,mask.sum()))

print("\n=== CHROMATIC RESIDUAL TEST ===")
print("band lambda Ccontrast Cdir Astd Npix")

for r in res:
    print(r)

lam=np.array([r[1] for r in res],float)

for k,label in [(2,"contrast"),(3,"directional")]:
    y=np.array([r[k] for r in res],float)
    coef=np.polyfit(np.log(lam),np.log(y),1)
    print()
    print(label,"alpha =",coef[0])
