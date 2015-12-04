#!/usr/bin/env python
import sys
import time
import os
import optparse
import numpy as np
import glob
from astropy.io import fits
import galsim
import metacal
import matplotlib.pyplot as plt

pixscale = 0.265
shear1_step = 0.01
shear2_step = 0.0
noise = 0.01

psf_im = galsim.fits.read("/u/ki/esheldon/tmp/for-eric/psf-000010-35.fits")
psf_im.scale = pixscale
psf_dil_im = metacal.getTargetPSF(psf_im,pixscale, g1=0.01)

image = galsim.fits.read("/u/ki/esheldon/tmp/for-eric/im-000010-35.fits")
image.scale = pixscale



image_noised = image.copy()
noiseModel = galsim.noise.GaussianNoise(sigma=noise)
image_noised.addNoise(noiseModel)
image_noised_orig = image_noised.copy()

# get the MetaCal images (without noise)
shearedGal, unshearedGal, reconv1PSF = metacal.metaCalibrate(image, psf_im,
                                                             g1 = shear1_step, g2 = shear2_step,
                                                             noise_symm = False, variance = noise**2)

# get the MetaCal images (without noise, without any noise modification)
shearedGal_noisy, unshearedGal_noisy,_ = metacal.metaCalibrate(image_noised, psf_im,
                                                             g1 = shear1_step, g2 = shear2_step,
                                                             noise_symm = False, variance = noise**2)
# get the MetaCal images (without noise, without any noise modification)
shearedGal_symm, unshearedGal_symm,_ = metacal.metaCalibrate(image_noised, psf_im,
                                                             g1 = shear1_step, g2 = shear2_step,
                                                             noise_symm = True, variance = noise**2)

noiseCorrImage, CNobj = metacal.getMetaCalNoiseCorrImage(image_noised, psf_im, psf_dil_im, g1 = shear1_step, g2=shear2_step, variance = noise**2)


pspec_noise = np.abs(np.fft.fftshift(np.fft.fft2((shearedGal_noisy - shearedGal).array)))**2
pspec_symm  = np.abs(np.fft.fftshift(np.fft.fft2((shearedGal_symm  - shearedGal_noisy).array)))**2



fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(nrows=2,ncols=3,figsize=(21,7))
plt1 = ax1.imshow(shearedGal.array,interpolation='nearest')
ax1.set_title("no-noise metacal")
plt2 = ax2.imshow(shearedGal_noisy.array,interpolation='nearest')
ax2.set_title("noisy metacal")
plt3 = ax3.imshow(shearedGal_symm.array,interpolation='nearest')
ax3.set_title("noise symm. metacal")

plt4 = ax4.imshow(pspec_noise,interpolation='nearest')
ax4.set_title("power spectrum of \n mcal_noise - mcal_nonoise")
plt5 = ax5.imshow(pspec_symm,interpolation='nearest')
ax5.set_title("power spectrum of \n mcal_symm - mcal_nonoise")
plt6 = ax6.imshow(noiseCorrImage.array,interpolation='nearest')
ax6.set_title("noise correlation function")

fig.colorbar(plt1,ax=ax1)
fig.colorbar(plt2,ax=ax2)
fig.colorbar(plt3,ax=ax3)
fig.colorbar(plt4,ax=ax4)
fig.colorbar(plt5,ax=ax5)
fig.colorbar(plt6,ax=ax6)

fig.savefig("metacal_noise_sheldon.png")

