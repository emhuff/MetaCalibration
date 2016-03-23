#!/usr/bin/env python
import sys
import time
import os
import optparse
import numpy as np
import glob
from astropy.io import fits
import galsim
#import matplotlib as mpl
#mpl.use('Agg')
import metacal
#import matplotlib.pyplot as plt




def metacal_noise_diagnose(e1_intrinsic = 0.0, e2_intrinsic = 0., shear1_step = 0.00, shear2_step = 0.00,
                           psf_size = .70, sersic_index = 4., pixscale = .265,
                           galaxy_size = 1.0, doplot = False, size = False,
                           do_centroid = False, noise = 0.01, getNoise = False):


    image_size = 64 #ceil(128 * (0.3/pixscale))
    psf_image_size = 48
    # We're worried about FFT accuracy, so there should be hooks here for the gsparams.
    gspars = galsim.GSParams()
    gspars.noise_pad_factor = 4*image_size
    gspars.noise_pad = noise**2
    
    # Create the undistorted galaxy, assign it some intrinsic ellipticity.
    obj = galsim.Sersic(sersic_index, half_light_radius = galaxy_size, flux=100.0, gsparams = gspars)
    obj_ellip = obj.lens(e1_intrinsic, e2_intrinsic, 1.)

    # define the psf.
    psf = galsim.Gaussian(half_light_radius= psf_size, gsparams = gspars)
            
    # Convolve with the PSF
    objConv = galsim.Convolve([psf,obj_ellip], gsparams = gspars)

    # Draw the fiducial noise-free data image.
    image = objConv.drawImage(image=galsim.Image(image_size,image_size,scale=pixscale), scale=pixscale )

    # Create the truth object we are trying to compare metacal to. First, shear the original Sersic object..
    obj_ellip_sheared  = obj_ellip.lens( shear1_step, shear2_step,1.0)
    
    # Then create the dilated psf.
    psf_dil = psf.dilate(1.+2*np.sqrt(shear1_step**2 + shear2_step**2))

    # Finally, convolve with the dilated psf to make the noise-free metacal truth image.
    objConv_sheared = galsim.Convolve([psf_dil,obj_ellip_sheared], gsparams = gspars)

    # And draw it into an image.
    image_sheared = objConv_sheared.drawImage(image=galsim.Image(image_size,image_size,scale=pixscale), scale=pixscale)
    
    # Make images of both psfs.
    psf_im = psf.drawImage(image=galsim.Image(psf_image_size,psf_image_size,scale=pixscale),scale=pixscale)
    psf_dil_im = psf_dil.drawImage(image=galsim.Image(psf_image_size,psf_image_size,scale=pixscale),scale=pixscale)
    
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
    shearedGal_noisy = metacal.metaCalibrateReconvolve(image_noised, psf, psf_dil,regularize= False,
                                                       g1=shear1_step, g2=shear2_step,
                                                       noise_symm = False, variance = noise**2)
    shearedGal_symm = metacal.metaCalibrateReconvolve(image_noised, psf, psf_dil,regularize= False,
                                                       g1=shear1_step, g2=shear2_step,
                                                       noise_symm = True, variance = noise**2)
    
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
    pspec_white = np.abs(np.fft.fftshift(np.fft.fft2((image_empty).array*(1./noise))))**2*(1./image.array.size)
    pspec_noise = np.abs(np.fft.fftshift(np.fft.fft2((shearedGal_noisy-image_sheared).array*(1./noise))))**2 *(1./image.array.size)
    pspec_symm = np.abs(np.fft.fftshift(np.fft.fft2((shearedGal_symm - image_sheared).array*(1./noise))))**2*(1./image.array.size)



    if doplot is True:
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        cmap = plt.cm.Greys
        # First plot: The images (true, metacal, difference):
        fig, ((ax1,ax2,ax3),(ax4,ax5,ax6),(ax7,ax8,ax9)) = plt.subplots(nrows=3,ncols=3,figsize=(20,20))
        plt1 = ax1.imshow(image_sheared.array,interpolation='nearest')
        ax1.set_title("'true' metacal image")
        plt2 = ax2.imshow(shearedGal.array,interpolation='nearest')
        ax2.set_title("metacal image")
        plt3 = ax3.imshow((shearedGal - image_sheared).array,interpolation='nearest')
        ax3.set_title("numerical error \n in metacal procedure")

        plt4 = ax4.imshow(image_noised.array,interpolation='nearest',cmap=cmap)
        ax4.set_title("noisy initial image")
        plt5 = ax5.imshow((shearedGal_noisy).array,interpolation='nearest',cmap=cmap)
        ax5.set_title("noisy metacal image")
        plt6 = ax6.imshow((shearedGal_symm).array,interpolation='nearest',cmap=cmap)
        ax6.set_title("symm.  image")

        plt7 = ax7.imshow((pspec_white),interpolation='nearest',cmap=plt.cm.winter)
        ax7.set_title("power spectrum of \n metacal -  truth images")
        plt8 = ax8.imshow((pspec_noise),interpolation='nearest',cmap=plt.cm.winter)
        ax8.set_title("power spectrum of \n mcal noise")    
        plt9 = ax9.imshow((pspec_symm),interpolation='nearest',cmap=plt.cm.winter)
        ax9.set_title(" power spectrum \n of isotropized noise")    




        fig.colorbar(plt1,ax=ax1)
        fig.colorbar(plt2,ax=ax2)
        fig.colorbar(plt3,ax=ax3)
        fig.colorbar(plt4,ax=ax4)
        fig.colorbar(plt5,ax=ax5)
        fig.colorbar(plt6,ax=ax6)
        fig.colorbar(plt7,ax=ax7)
        fig.colorbar(plt8,ax=ax8)
        fig.colorbar(plt9,ax=ax9)
        fig.savefig("metacal_noise_diagnostics.png")
        fig.clf()
    if getNoise is False:
        return status, res_noise.corrected_e1, res_symm.corrected_e1, res_nonoise.corrected_e1, res_white.corrected_e1
    else:
        return pspec_white, pspec_noise, pspec_symm

def main(argv):
    n_iter = 10000
    shear1_step = 0.001
    shear2_step = 0.0
    e1_intrinsic = 0.3
    e2_intrinsic = 0.0
    noise = 0.4
    getNoise = True
    Enoise1 = []
    Esymm1 = []
    Etrue1 = []
    Ewhite1 = []
    Enoise2 = []
    Esymm2 = []
    Etrue2 = []
    Ewhite2 = []    
    #(status, this_Enoise, this_Esymm, this_Etrue, this_Ewhite) = metacal_noise_diagnose(e1_intrinsic = e1_intrinsic, e2_intrinsic = e2_intrinsic,
    #                                                                                    shear1_step = shear1_step, shear2_step = shear2_step,
    #                                                                                    doplot=True,noise= noise)


    for i in xrange(n_iter):
        #print "iter "+str(i)+" of "+str(n_iter)
        if getNoise is False:
            (status1, this_Enoise1, this_Esymm1, this_Etrue1, this_Ewhite1) = metacal_noise_diagnose(e1_intrinsic = e1_intrinsic, e2_intrinsic = e2_intrinsic,
                                                                        shear1_step = shear1_step, shear2_step = shear2_step, doplot=False,noise= noise)
            (status2, this_Enoise2, this_Esymm2, this_Etrue2, this_Ewhite2) = metacal_noise_diagnose(e1_intrinsic = e1_intrinsic, e2_intrinsic = e2_intrinsic,
                                                                        shear1_step = -shear1_step, shear2_step = -shear2_step, doplot=False,noise= noise)
            if (status1 is True) & (status2 is True):
                Enoise1.append(this_Enoise1)
                Esymm1.append(this_Esymm1)
                Etrue1.append(this_Etrue1)
                Ewhite1.append(this_Ewhite1)
                Enoise2.append(this_Enoise2)
                Esymm2.append(this_Esymm2)
                Etrue2.append(this_Etrue2)
                Ewhite2.append(this_Ewhite2)
        else:
            if i == 0:
                noisePower_white, noisePower_mcal, noisePower_symm = metacal_noise_diagnose(e1_intrinsic = e1_intrinsic, e2_intrinsic = e2_intrinsic,
                                                                                             shear1_step = shear1_step, shear2_step = shear2_step,
                                                                                             doplot=False,noise= noise,getNoise = True)
            else:
                thisPower_white, thisPower_mcal, thisPower_symm = metacal_noise_diagnose(e1_intrinsic = e1_intrinsic, e2_intrinsic = e2_intrinsic,
                                                                        shear1_step = shear1_step, shear2_step = shear2_step, doplot=False,noise= noise,
                                                                        getNoise = True)

                noisePower_white = noisePower_white + thisPower_white
                noisePower_mcal = noisePower_mcal + thisPower_mcal
                noisePower_symm = noisePower_symm + thisPower_symm
    if getNoise is False:        
        Enoise1 = np.array(Enoise1)
        Esymm1 = np.array(Esymm1)
        Etrue1 = np.array(Etrue1)
        Ewhite1 = np.array(Ewhite1)
        Enoise2 = np.array(Enoise2)
        Esymm2 = np.array(Esymm2)
        Etrue2 = np.array(Etrue2)
        Ewhite2 = np.array(Ewhite2)    
        image_size = 48
        obj = galsim.Sersic(4, half_light_radius =1.0, flux=100.0)
        objEllip = obj.lens(e1_intrinsic, e2_intrinsic, 1.)

        # Convolve with a gaussian PSF
        psf = galsim.Moffat(fwhm = 0.7, beta=3.5)
        objConv = galsim.Convolve([psf,objEllip])
        image = objConv.drawImage(image=galsim.Image(image_size,image_size,scale=0.265) )
    
    
        #print "at S/N=",np.sqrt(np.sum(image.array**2)/noise**2)
        #print "truth, white noise", np.mean(Ewhite-Etrue),"+/-",np.std(Ewhite-Etrue)/np.sqrt(n_iter)
        #print "metacal only: ",np.mean(Enoise-Ewhite)," +/- ",np.std(Enoise - Ewhite)/np.sqrt(n_iter)
        #print "metacal with symm: ",np.mean(Esymm-Ewhite)," +/- ",np.std(Esymm - Etrue)/np.sqrt(n_iter)
        print np.mean(Ewhite1-Etrue1), np.mean(Enoise1-Ewhite1), np.mean(Esymm1-Ewhite1), \
        np.mean(Ewhite2-Etrue2), np.mean(Enoise2-Ewhite2), np.mean(Esymm2-Ewhite2)
    else:
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        cmap = plt.cm.Greys

        noisePower_mcal = noisePower_mcal * 1./n_iter
        noisePower_symm = noisePower_symm * 1./n_iter
        fig,((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(nrows=2,ncols=3,figsize=(21,7))
        plt1 = ax1.imshow(noisePower_white,interpolation='nearest')
        ax1.set_title("averaged white noise power,\n"+str(n_iter)+" iterations")
        plt2 = ax2.imshow(noisePower_mcal,interpolation='nearest')
        ax2.set_title("averaged mcal noise power,\n"+str(n_iter)+" iterations")
        plt3 = ax3.imshow(noisePower_symm,interpolation='nearest')
        ax3.set_title("averaged mcal symm. noise power,\n"+str(n_iter)+" iterations")

        plt4 = ax4.imshow(noisePower_white - noisePower_white.transpose(),interpolation='nearest')
        plt5 = ax5.imshow(noisePower_mcal - noisePower_mcal.transpose(),interpolation='nearest')
        plt6 = ax6.imshow(noisePower_symm - noisePower_symm.transpose(),interpolation='nearest')                
        fig.colorbar(plt2,ax=ax1)
        fig.colorbar(plt2,ax=ax2)
        fig.colorbar(plt2,ax=ax3)
        fig.colorbar(plt2,ax=ax4)
        fig.colorbar(plt2,ax=ax5)
        fig.colorbar(plt2,ax=ax6)
        fig.savefig("mcal_avg_noise_power.png")
        
if __name__ == "__main__":
    import pdb, traceback
    try:
        main(sys.argv)
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

