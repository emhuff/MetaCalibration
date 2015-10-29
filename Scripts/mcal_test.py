#!/usr/bin/env python
import sys
import os
import numpy as np
import esutil
import matplotlib as mpl
mpl.use('Agg')
import MetaCalGreat3Wrapper as mcg3
import galsim



def getCases(path='../Data/mcal-tests/'):
    case_g1 = {'g1':0.01,'g2':0.0,'psf_e1':0.0,'psf_e2':0.0,
               'galImage':path+'test-image-mcal-galshear-0.01-0.00.fits',
               'psfImage':path+'test-psf-mcal-galshear-0.01-0.00.fits'}

    case_g2 = {'g1':0.00,'g2':0.01,'psf_e1':0.0,'psf_e2':0.0,
               'galImage':path+'test-image-mcal-galshear-0.00-0.01.fits',
               'psfImage':path+'test-psf-mcal-galshear-0.00-0.01.fits'}

    case_psf_e1 = {'g1':0.00,'g2':0.00,'psf_e1':0.01,'psf_e2':0.0,
                   'galImage':path+'test-image-mcal-psfshear-0.01-0.00.fits',
                   'psfImage':path+'test-psf-mcal-psfshear-0.01-0.00.fits'}

    case_psf_e2 = {'g1':0.00,'g2':0.00,'psf_e1':0.0,'psf_e2':0.01,
                   'galImage':path+'test-image-mcal-psfshear-0.00-0.01.fits',
                   'psfImage':path+'test-psf-mcal-psfshear-0.00-0.01.fits'}

    cases = [case_g1,case_g2,case_psf_e1,case_psf_e2]
    return cases


def getDiffImages(case,path='../Data/mcal-tests/'):
    gal_im = galsim.fits.read(path+"test-image.fits")
    psf_im = galsim.fits.read(path+"test-psf.fits")
    galSheldon  = galsim.fits.read(case['galImage'])
    psfSheldon = galsim.fits.read(case['psfImage'])
    if (case['psf_e1'] !=0) | (case['psf_e2'] !=0):
        gal_shear=False
    else:
        gal_shear=True
    shearedGalaxy, unshearedGalaxy, reconvPSF = mcg3.metaCalibrate(gal_im, psf_im, 
                                                                   g1 = case['g1'], 
                                                                   g2 = case['g2'], 
                                                                   gal_shear=gal_shear)
    outDiffGal = shearedGalaxy - galSheldon 
    outDiffPSF = reconvPSF - psfSheldon
    print np.abs(outDiffGal.array).max()/galSheldon.array.max(),np.abs(outDiffPSF.array).max()/psfSheldon.array.max()
    return outDiffGal, outDiffPSF, galSheldon, psfSheldon

def doPlot(case):
    gal_diff, psf_diff,gal_sheldon, psf_sheldon = getDiffImages(case)
    import matplotlib.pyplot as plt
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(14,6.5))
    plt1 = ax1.imshow(gal_diff.array/np.max(gal_sheldon.array),vmin = -.005, vmax = 0.005, cmap = plt.cm.bwr)
    ax1.set_title('huff - sheldon, galaxy\n '+case['galImage'])
    plt2 = ax2.imshow(psf_diff.array/np.max(psf_sheldon.array),vmin = -.005, vmax = 0.005, cmap = plt.cm.bwr)
    ax2.set_title('huff - sheldon, psf\n '+case['psfImage'])

    plotFile = os.path.splitext(case['galImage'])[0]
    fig.colorbar(plt1,ax=ax1)
    fig.colorbar(plt2,ax=ax2)
    fig.savefig(plotFile+'.png')

def main(argv):
    cases = getCases()
    for case in cases:
        doPlot(case)
        
    
    
if __name__ == "__main__":
    import pdb, traceback
    try:
        main(sys.argv)
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
