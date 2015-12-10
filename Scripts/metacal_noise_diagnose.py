#!/usr/bin/env python
import sys
import time
import os
import optparse
import numpy as np
import glob
from astropy.io import fits
import galsim
import matplotlib as mpl
mpl.use('Agg')
import metacal
import matplotlib.pyplot as plt




def metacal_noise_diagnose(e1_intrinsic = 0.0, e2_intrinsic = 0., shear1_step = 0.00, shear2_step = 0.,
                           psf_size = 0.7, sersic_index = 4., pixscale = 0.265,
                           galaxy_size = 1.0, doplot = False, size = False,
                           do_centroid = False, noise = 0.01):


    image_size = 64#ceil(128 * (0.3/pixscale))
    psf_image_size = 64
    # We're worried about FFT accuracy, so there should be hooks here for the gsparams.
    gspars = galsim.GSParams()
    
    # Create the undistorted galaxy, assign it some intrinsic ellipticity.
    obj = galsim.Sersic(sersic_index, half_light_radius =galaxy_size, flux=100.0, gsparams = gspars)
    objEllip = obj.lens(e1_intrinsic, e2_intrinsic, 1.)

    # Convolve with a gaussian PSF
    #psf = galsim.Gaussian(sigma= psf_size, gsparams = gspars)
    psf = galsim.Moffat(fwhm = psf_size, beta=3.5)
    objConv = galsim.Convolve([psf,objEllip], gsparams = gspars)
    image = objConv.drawImage(image=galsim.Image(image_size,image_size,scale=pixscale) )

    # Create the truth object to compare the metacalibration-generated image to.
    objEllip_sheared  = objEllip.lens( shear1_step, shear2_step,1.0)
    psf_dil = psf.dilate(1.+2*np.sqrt(shear1_step**2 + shear2_step**2))
    objConv_sheared = galsim.Convolve([psf_dil,objEllip_sheared], gsparams = gspars)
    image_sheared = objConv_sheared.drawImage(image=galsim.Image(image_size,image_size,scale=pixscale) )

    # Convolve our psf objects with the pixel.
    pixelObj = galsim.Pixel(scale=pixscale)
    psf = galsim.Convolve([psf,pixelObj])
    psf_dil = galsim.Convolve([psf_dil,pixelObj])

    
    # Make an image of the psf (the dilated one, which we use for the measurement image)
    psf_im = psf.drawImage(image=galsim.Image(psf_image_size,psf_image_size,scale=pixscale), method='no_pixel' )
    psf_dil_im = psf_dil.drawImage(image=galsim.Image(psf_image_size,psf_image_size,scale=pixscale), method='no_pixel' )

    #psf = galsim.InterpolatedImage(psf_im)
    #psf_dil = psf.dilate(1.+2*np.sqrt(shear1_step**2 + shear2_step**2))
    
    # Add noise to the image.

    # make an empty image with this noise
    image_empty = galsim.Image(image.bounds,scale=image.scale)
    noiseModel = galsim.noise.GaussianNoise(sigma=noise)
    image_empty.addNoise(noiseModel)
    image_noised = image_empty + image
    image_sheared_noised = image_empty + image_sheared
    

    shearedGal = metacal.metaCalibrateReconvolve(image, psf, psf_dil,
                                                 g1=shear1_step, g2=shear2_step,
                                                 noise_symm = False, variance = noise**2)
    shearedGal_noisy = metacal.metaCalibrateReconvolve(image_noised, psf, psf_dil,
                                                       g1=shear1_step, g2=shear2_step,
                                                       noise_symm = False, variance = noise**2)
    shearedGal_empty = metacal.metaCalibrateReconvolve(image_empty, psf, psf_dil,
                                                       g1=shear1_step, g2=shear2_step,
                                                       noise_symm = False, variance = noise**2)
    shearedGal_symm = metacal.metaCalibrateReconvolve(image_noised, psf, psf_dil,
                                                       g1=shear1_step, g2=shear2_step,
                                                       noise_symm = True, variance = noise**2)

    #shearedGal_noisy =  image_empty + image_sheared
    #shearedGal_symm = shearedGal_empty + image_sheared
    
    res_nonoise = galsim.hsm.EstimateShear(image_sheared, psf_dil_im, sky_var= noise**2,strict=False)
    res_white   = galsim.hsm.EstimateShear(image_sheared_noised, psf_dil_im, sky_var= noise**2,strict=False)
    res_noise   = galsim.hsm.EstimateShear(shearedGal_noisy, psf_dil_im, sky_var= noise**2,strict=False)
    res_symm   = galsim.hsm.EstimateShear(shearedGal_symm, psf_dil_im, sky_var= noise**2,strict=False)
    if (res_nonoise.error_message == "") & (res_white.error_message == "") & (res_noise.error_message == "") & (res_symm.error_message == ""):
        status = True
    else:
        status = False
    #print "shapes:"
    #print "with noise:", res_noise.corrected_e1
    #print "with symm:", res_symm.corrected_e1
    #print "no noise:",res_nonoise.corrected_e1
    
    # Get the MetaCal noise correlation function image.
    pspec_noisy = np.abs(np.fft.fftshift(np.fft.fft2((shearedGal_noisy-image_noised).array)))**2
    pspec_orig = np.abs(np.fft.fftshift(np.fft.fft2((image_sheared-image).array*(1./noise))))**2
    pspec_mcal = np.abs(np.fft.fftshift(np.fft.fft2((shearedGal - image).array*(1./noise))))**2



    
    # First plot: The images (true, metacal, difference):
    fig, ((ax1,ax2,ax3),(ax4,ax5,ax6),(ax7,ax8,ax9)) = plt.subplots(nrows=3,ncols=3,figsize=(20,20))
    plt1 = ax1.imshow(image_sheared.array,interpolation='nearest')
    ax1.set_title("'true' metacal image")
    plt2 = ax2.imshow(shearedGal.array,interpolation='nearest')
    ax2.set_title("metacal image")
    plt3 = ax3.imshow((shearedGal - image_sheared).array,interpolation='nearest')
    ax3.set_title("numerical error \n in metacal procedure")

    plt4 = ax4.imshow(image_noised.array,interpolation='nearest')
    ax4.set_title("noisy initial image")
    plt5 = ax5.imshow((shearedGal_noisy).array,interpolation='nearest')
    ax5.set_title("noisy metacal image")
    #plt6 = ax6.imshow(noiseCorrImage.array,interpolation='nearest')
    #ax6.set_title("2d noise \n correlation function")

    plt7 = ax7.imshow(np.log10(pspec_orig),interpolation='nearest')
    ax7.set_title("log_10 of power spectrum of \n before - after truth images")

    plt8 = ax8.imshow(np.log10(pspec_mcal),interpolation='nearest')
    ax8.set_title("log_10 of power spectrum of \n before - after noiseless mcal images")    
    plt9 = ax9.imshow(np.log10(pspec_noisy),interpolation='nearest')
    ax9.set_title("log_10 of power spectrum of \n before - after noisy mcal images")    

    

    print "initial noise:",np.std(image_noised.array - image.array)
    #print "estimated noise after noise symmetrization processing:", np.sqrt(CNobj.getVariance())
    print "actual noise after processing::",np.std(shearedGal_noisy.array - image_sheared.array)


    

    fig.colorbar(plt1,ax=ax1)
    fig.colorbar(plt2,ax=ax2)
    fig.colorbar(plt3,ax=ax3)
    fig.colorbar(plt4,ax=ax4)
    fig.colorbar(plt5,ax=ax5)
    #fig.colorbar(plt6,ax=ax6)
    fig.colorbar(plt7,ax=ax7)
    fig.colorbar(plt8,ax=ax8)
    fig.colorbar(plt9,ax=ax9)
    fig.savefig("metacal_noise_images.png")
    fig.clf()
    return status, res_noise.corrected_e1, res_symm.corrected_e1, res_nonoise.corrected_e1, res_white.corrected_e1


def main(argv):
    npts = 20
    n_iter = 1000
    e_arr =  np.linspace(-0.5, 0.5, npts)
    R_true_arr = e_arr*0.
    R_est_arr = e_arr*0.
    R_sig_arr = e_arr * 0.
    R_rec_arr = e_arr*0.
    shear1_step = 0.001
    shear2_step = 0.0
    e1_intrinsic = 0.0
    e2_intrinsic = 0.
    noise = 0.4
    Enoise = []
    Esymm= []
    Etrue = []
    Ewhite = []
    (status, this_Enoise, this_Esymm, this_Etrue, this_Ewhite) = metacal_noise_diagnose(e1_intrinsic = e1_intrinsic, e2_intrinsic = e2_intrinsic,
                                                                                        shear1_step = shear1_step, shear2_step = shear2_step,
                                                                                        doplot=True,noise= noise)


    for i in xrange(n_iter):
        print "iter "+str(i)+" of "+str(n_iter)
        (status, this_Enoise, this_Esymm, this_Etrue, this_Ewhite) = metacal_noise_diagnose(e1_intrinsic = e1_intrinsic, e2_intrinsic = e2_intrinsic,
                                                                       shear1_step = shear1_step, shear2_step = shear2_step, doplot=False,noise= noise)
        if status is True:
            Enoise.append(this_Enoise)
            Esymm.append(this_Esymm)
            Etrue.append(this_Etrue)
            Ewhite.append(this_Ewhite)
            Enoise_arr = np.array(Enoise)
            Esymm_arr = np.array(Esymm)
            Etrue_arr = np.array(Etrue)
            Ewhite_arr = np.array(Ewhite)
            print "truth, white noise", np.mean(Ewhite_arr-Etrue_arr),"+/-",np.std(Ewhite_arr-Etrue_arr)/np.sqrt(i+1)
            print "metacal only: ",np.mean(Enoise_arr-Ewhite_arr)," +/- ",np.std(Enoise_arr - Ewhite_arr)/np.sqrt(i+1)
            print "metacal with symm: ",np.mean(Esymm_arr-Ewhite_arr)," +/- ",np.std(Esymm_arr - Ewhite_arr)/np.sqrt(i+1)
        else:
            print "shape measurement failed."



        
    Enoise = np.array(Enoise)
    Esymm = np.array(Esymm)
    Etrue = np.array(Etrue)
    Ewhite = np.array(Ewhite)
    image_size = 48
    obj = galsim.Sersic(4, half_light_radius =1.0, flux=100.0)
    objEllip = obj.lens(e1_intrinsic, e2_intrinsic, 1.)

    # Convolve with a gaussian PSF
    psf = galsim.Moffat(fwhm = 0.7, beta=3.5)
    objConv = galsim.Convolve([psf,objEllip])
    image = objConv.drawImage(image=galsim.Image(image_size,image_size,scale=0.265) )

    
    print "at S/N=",np.sqrt(np.sum(image.array**2)/noise**2)
    print "truth, white noise", np.mean(Ewhite-Etrue),"+/-",np.std(Ewhite-Etrue)/np.sqrt(n_iter)
    print "metacal only: ",np.mean(Enoise-Ewhite)," +/- ",np.std(Enoise - Ewhite)/np.sqrt(n_iter)
    print "metacal with symm: ",np.mean(Esymm-Ewhite)," +/- ",np.std(Esymm - Etrue)/np.sqrt(n_iter)
    stop

if __name__ == "__main__":
    import pdb, traceback
    try:
        main(sys.argv)
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

