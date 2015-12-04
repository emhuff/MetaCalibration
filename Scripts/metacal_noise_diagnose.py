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


def centroid(image= None, weight = None):

    shape = image.array.shape

    xx,yy = np.meshgrid( np.arange(shape[0]), np.arange(shape[1]) )

    xcen = np.sum(image.array*xx)*1./np.sum(image.array)
    ycen = np.sum(image.array*yy)*1./np.sum(image.array)
    return xcen, ycen


def size_mom(image= None, weight = None):

    shape = image.array.shape
    xcen, ycen = centroid(image=image, weight= weight)
    
    
    xx,yy = np.meshgrid( np.arange(shape[0]) - xcen, np.arange(shape[1]) - ycen )

    x2 = np.sum(image.array*xx**2)*1./np.sum(image.array)
    y2 = np.sum(image.array*yy**2)*1./np.sum(image.array)

    
    
    return x2,y2


def metacal_noise_diagnose(e1_intrinsic = 0.0, e2_intrinsic = 0., shear1_step = 0.00, shear2_step = 0., psf_size =
                     1.0, sersic_index = 4., pixscale = 0.2,
                     galaxy_size = 2.50, doplot = False, size = False,
                     do_centroid = False, noise = 0.01):


    image_size = np.ceil(128 * (0.3/pixscale))
    psf_image_size = 51
    
    # We're worried about FFT accuracy, so there should be hooks here for the gsparams.
    gspars = galsim.GSParams()
    
    # Create the undistorted galaxy, assign it some intrinsic ellipticity.
    obj = galsim.Sersic(sersic_index, half_light_radius =galaxy_size, flux=100.0, gsparams = gspars)
    objEllip = obj.lens(e1_intrinsic, e2_intrinsic, 1.)

    # Convolve with a gaussian PSF
    psf = galsim.Gaussian(sigma= psf_size, gsparams = gspars)
    objConv = galsim.Convolve([psf,objEllip], gsparams = gspars)
    image = objConv.drawImage(image=galsim.Image(image_size,image_size,scale=pixscale) )

    # Create the truth object to compare the metacalibration-generated image to.
    objEllip_sheared  = objEllip.lens( shear1_step, shear2_step,1.0)
    psf_dil = galsim.Gaussian(sigma=psf_size*(1+2*np.sqrt(shear1_step**2 + shear2_step**2)) , gsparams = gspars)
    objConv_sheared = galsim.Convolve([psf_dil,objEllip_sheared], gsparams = gspars)
    image_sheared = objConv_sheared.drawImage(image=galsim.Image(image_size,image_size,scale=pixscale) )
    
    # Make an image of the psf (the dilated one, which we use for the measurement image)
    psf_im = psf.drawImage(image=galsim.Image(psf_image_size,psf_image_size,scale=pixscale) )
    psf_dil_im = psf_dil.drawImage(image=galsim.Image(psf_image_size,psf_image_size,scale=pixscale) )

    # Add noise to the image.
    image_noised = image.copy()
    noiseModel = galsim.noise.GaussianNoise(sigma=noise)
    image_noised.addNoise(noiseModel)
    image_noised_orig = image_noised.copy()

    psf = metacal.getTargetPSF(psf_im, pixscale, g1 =0.0, g2 = 0.0, gal_shear=True)
    plt.imshow((psf - psf_dil_im).array)
    plt.show()
    print np.min((psf - psf_dil_im).array),np.max((psf - psf_dil_im).array)
    stop

    
    # get the MetaCal images (without noise)
    shearedGal, unshearedGal, reconv1PSF = metacal.metaCalibrate(image, psf_im,#targetPSFImage = psf_dil_im,
                                                                   g1 = shear1_step, g2 = shear2_step,
                                                                   noise_symm = False, variance = noise**2)
    # get the MetaCal images (with noise)
    shearedGal_noisy, unshearedGal_noisy, _ = metacal.metaCalibrate(image_noised, psf_im,#targetPSFImage = psf_dil_im,
                                                                            g1 = shear1_step, g2 = shear2_step,
                                                                            noise_symm = False, variance = noise**2)
    # Get the MetaCal noise correlation function image.
    noiseCorrImage, CNobj = metacal.getMetaCalNoiseCorrImage(image_noised, psf_im, psf_dil_im, g1 = shear1_step, g2=shear2_step, variance = noise**2)
    pspec_noisy = np.abs(np.fft.fftshift(np.fft.fft2((shearedGal_noisy-image_noised).array)))**2
    pspec_orig = np.abs(np.fft.fftshift(np.fft.fft2((image_sheared-image).array)))**2
    pspec_mcal = np.abs(np.fft.fftshift(np.fft.fft2((shearedGal - image).array)))**2

    # factor of 10 in power is a sort-of-arbitrary threshold... but looking at plots of the power spectra suggests that it's still a fair place to start.
    bad = np.abs(np.fft.fft2(((shearedGal_noisy-image_noised).array)))**2 > 10
    image_reconv_fft = np.fft.fft2(shearedGal_noisy.array)
    image_reconv_fft[bad] = 0.
    image_reconv_better= np.real(np.fft.ifft2(image_reconv_fft))
    shearedGal_noisy_better = galsim.Image(np.ascontiguousarray(image_reconv_better), scale=shearedGal_noisy.scale)


    
    # First plot: The images (true, metacal, difference):
    fig, ((ax1,ax2,ax3),(ax4,ax5,ax6),(ax7,ax8,ax9)) = plt.subplots(nrows=3,ncols=3,figsize=(20,20))
    plt1 = ax1.imshow(image_sheared.array)
    ax1.set_title("'true' metacal image")
    plt2 = ax2.imshow(shearedGal.array)
    ax2.set_title("metacal image")
    plt3 = ax3.imshow((shearedGal - image_sheared).array)
    ax3.set_title("numerical error \n in metacal procedure")

    plt4 = ax4.imshow(image_noised_orig.array)
    ax4.set_title("noisy initial image")
    plt5 = ax5.imshow((shearedGal_noisy).array)
    ax5.set_title("noisy metacal image")
    plt6 = ax6.imshow(noiseCorrImage.array)
    ax6.set_title("2d noise \n correlation function")

    plt7 = ax7.imshow(np.log10(pspec_orig))
    ax7.set_title("log_10 of power spectrum of \n before - after truth images")

    plt8 = ax8.imshow(np.log10(pspec_mcal))
    ax8.set_title("log_10 of power spectrum of \n before - after noiseless mcal images")    
    plt9 = ax9.imshow(np.log10(pspec_noisy))
    ax9.set_title("log_10 of power spectrum of \n before - after noisy mcal images")    

    

    print "initial noise:",np.std(image_noised.array - image.array)
    print "estimated noise after noise symmetrization processing:", np.sqrt(CNobj.getVariance())
    print "actual noise after noise symmetrization processing::",np.std(shearedGal_noisy.array - image_sheared.array)
    print "noise in cleaned image is:",np.std(shearedGal_noisy_better.array - image_sheared.array)

    

    fig.colorbar(plt1,ax=ax1)
    fig.colorbar(plt2,ax=ax2)
    fig.colorbar(plt3,ax=ax3)
    fig.colorbar(plt4,ax=ax4)
    fig.colorbar(plt5,ax=ax5)
    fig.colorbar(plt6,ax=ax6)
    fig.colorbar(plt7,ax=ax7)
    fig.colorbar(plt8,ax=ax8)
    fig.colorbar(plt9,ax=ax9)
    fig.savefig("metacal_noise_images.png")
    stop

def main(argv):
    npts = 20
    n_iter = 50
    e_arr =  np.linspace(-0.5, 0.5, npts)
    R_true_arr = e_arr*0.
    R_est_arr = e_arr*0.
    R_sig_arr = e_arr * 0.
    R_rec_arr = e_arr*0.
    shear1_step = 0.01
    shear2_step = 0.0
    e1_intrinsic = 0.0
    e2_intrinsic = 0.
    noise = .01
    
    metacal_noise_diagnose(e1_intrinsic = e1_intrinsic, e2_intrinsic = e2_intrinsic, shear1_step = shear1_step, shear2_step = shear2_step, doplot=True,noise= noise)

    

if __name__ == "__main__":
    import pdb, traceback
    try:
        main(sys.argv)
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

