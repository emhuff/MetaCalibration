#!/usr/bin/env python
import sys
import time
import os
import optparse
import numpy as np
import glob
from astropy.io import fits
import galsim
import MetaCalGreat3Wrapper as mcG3
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


def metacal_diagnose(e1_intrinsic = 0.0, e2_intrinsic = 0., shear1_step = 0.01, shear2_step = 0., psf_size =
                     1.0, sersic_index = 4., pixscale = 0.3,
                     galaxy_size = 2.0, doplot = False, size = False,
                     do_centroid = False):


    image_size = np.ceil(125 * (0.3/pixscale))
    
    # We're worried about FFT accuracy, so there should be hooks here for the gsparams.
    gspars = galsim.GSParams()
    
    # Create the undistorted galaxy, assign it a large intrinsic ellipticity.
    obj = galsim.Sersic(sersic_index, half_light_radius =galaxy_size, flux=1.0, gsparams = gspars)
    objEllip = obj.lens(e1_intrinsic, e2_intrinsic, 1.)


    # Convolve with a gaussian PSF
    psf = galsim.Gaussian(sigma= psf_size, gsparams = gspars)
    objConv = galsim.Convolve([psf,objEllip], gsparams = gspars)
    image = objConv.drawImage(image=galsim.Image(image_size,image_size,scale=pixscale) )

    # Create the object to compare the metacalibration-generated image to.
    objEllip2 = objEllip.lens( shear1_step, shear2_step,1.0)
    objConv2 = galsim.Convolve([psf,objEllip2], gsparams = gspars)
    image2 = objConv2.drawImage(image=galsim.Image(image_size,image_size,scale=pixscale) )

    # Make an image of the psf
    psf_im = psf.drawImage(image=galsim.Image(image_size,image_size,scale=pixscale) )

    # Copied straight from MetaCalGreat3Wrapper.py
    sheared1Galaxy, unsheared1Galaxy, reconv1PSF = mcG3.metaCalibrate(image, psf_im, g1 = shear1_step, g2 = shear2_step)
    shearedm1Galaxy, unshearedm1Galaxy, reconvm1PSF = mcG3.metaCalibrate(image, psf_im, g1 =  -shear1_step, g2 = - shear2_step)

    # Make an interpolated image of the psf.
    l5 = galsim.Lanczos(5, True, 1.0E-4)
    l52d = galsim.InterpolantXY(l5)
    reconvPSFObj = galsim.InterpolatedImage(reconv1PSF, gsparams = gspars)

    # These objects use the original galaxy convolved with the metacal-dilated
    # psf to measure what response we ought to be getting from metacal
    # Make a dilated image of the psf
    psf_dil = galsim.Gaussian(sigma=psf_size*(1+2*np.sqrt(shear1_step**2 + shear2_step**2)) , gsparams = gspars)
    psf_im_dil = psf_dil.drawImage(image=galsim.Image(25,25,scale=pixscale) )

    objConv3 = galsim.Convolve([psf_dil,objEllip], gsparams = gspars)
    image3 = objConv3.drawImage(image=galsim.Image(image_size,image_size,scale=pixscale) )

    objConv4 = galsim.Convolve([psf_dil,objEllip2], gsparams = gspars)
    image4 = objConv4.drawImage(image=galsim.Image(image_size,image_size,scale=pixscale) )


    if doplot is True:

        fig,(ax1, ax2, ax3) = plt.subplots(nrows=1,ncols=3,figsize=(21,6))
        plt1 = ax1.imshow(image4.array)
        ax1.set_title("(true) sheared image")
        plt2 = ax2.imshow(sheared1Galaxy.array)
        ax2.set_title("metacal image")
        plt3 = ax3.imshow((sheared1Galaxy.array -  image4.array)/0.001 , vmin = -.001, vmax = 0.001, cmap = plt.cm.bwr)
        ax3.set_title("difference / 0.001")
        fig.colorbar(plt1,ax=ax1)
        fig.colorbar(plt2,ax=ax2)
        fig.colorbar(plt3,ax=ax3)
        plt.show()
        '''
        fig,(ax1, ax2, ax3) = plt.subplots(nrows=1,ncols=3,figsize=(21,7))
        plt1 = ax1.imshow(psf_im_dil.array)
        ax1.set_title("(true) sheared image")
        plt2 = ax2.imshow(reconv1PSF.array)
        ax2.set_title("metacal image")
        plt3 = ax3.imshow((reconv1PSF.array -  psf_im_dil.array)/0.001 , vmin = -.01, vmax = 0.01, cmap = plt.cm.bwr)
        ax3.set_title("difference / 0.001")
        fig.colorbar(plt1,ax=ax1)
        fig.colorbar(plt2,ax=ax2)
        fig.colorbar(plt3,ax=ax3)
        plt.show()
        '''

    # Now measure the shapes of the metacal galaxies.
    res_g1  = galsim.hsm.EstimateShear(sheared1Galaxy,  reconv1PSF,  guess_sig_PSF = 1.0, shear_est="regauss")
    res_mg1 = galsim.hsm.EstimateShear(shearedm1Galaxy, reconvm1PSF ,guess_sig_PSF = 1.0, shear_est="regauss")
    

    # Then measure the change in shape of the original galaxy, when the
    # true shear is changed.
    shape_0 = galsim.hsm.EstimateShear(image,psf_im, guess_sig_PSF = 1.0, shear_est = "regauss")
    shape_1 = galsim.hsm.EstimateShear(image2,psf_im,guess_sig_PSF = 1.0, shear_est = "regauss")
    
    # Finally, do the same thing with the slightly dilated psf used by metacal.
    shape_3 = galsim.hsm.EstimateShear(image3,reconv1PSF, guess_sig_PSF = 1.0, shear_est = "regauss")
    shape_4 = galsim.hsm.EstimateShear(image4,reconv1PSF, guess_sig_PSF = 1.0, shear_est = "regauss")
    
    # The first two of these three shape measurements should agree!
    if shear1_step != 0. and shear2_step == 0.:

        de1_g1_est = 0.5*(res_g1.corrected_e1 - res_mg1.corrected_e1)/shear1_step
        de1_g1_rec = (shape_4.corrected_e1 - shape_3.corrected_e1)/shear1_step
        de1_g1_tru = (shape_1.corrected_e1 - shape_0.corrected_e1)/shear1_step
        print "True R1:", de1_g1_tru
        print "Estimated R1:", de1_g1_est
        print "True Reconv R1", de1_g1_rec
            
    # The first two of these three shape measurements should agree!
    if shear2_step != 0. and shear1_step == 0.:
        de2_g2_est = 0.5*(res_g1.corrected_e2 - res_mg1.corrected_e2)/shear2_step
        de2_g2_rec = (shape_4.corrected_e2 - shape_3.corrected_e2)/shear2_step
        de2_g2_tru = (shape_1.corrected_e2 - shape_0.corrected_e2)/shear2_step
        print "True R2:", de2_g2_tru
        print "Estimated R2:", de2_g2_est
        print "True Reconv R2", de2_g2_rec
    if size is True:
        # Now for some other diagnostics. Do our image centroids shift at
        # all?
        x2_0, y2_0 = size_mom(image)
        x2_1, y2_1 = size_mom(image2)
        x2_mcp, y2_mcp = size_mom(sheared1Galaxy)
        x2_mcm, y2_mcm = size_mom(shearedm1Galaxy)
        x2_4, y2_4 = size_mom(image3)
        x2_5, y2_5 = size_mom(image4)

        r0 = image.FindAdaptiveMom().moments_sigma#np.sqrt(x2_0 + y2_0)
        r1 = image2.FindAdaptiveMom().moments_sigma#np.sqrt(x2_1 + y2_1)
        r_mcp = sheared1Galaxy.FindAdaptiveMom().moments_sigma#np.sqrt(x2_mcp + y2_mcp)
        r_mcm = shearedm1Galaxy.FindAdaptiveMom().moments_sigma#np.sqrt(x2_mcm + y2_mcm)
        r_4 = image3.FindAdaptiveMom().moments_sigma#np.sqrt(x2_4 + y2_4)
        r_5 = image4.FindAdaptiveMom().moments_sigma#np.sqrt(x2_5 + y2_5)
        return r1, r_mcp, r_5    

    if do_centroid is True:
        # Now for some other diagnostics. Do our image centroids shift at
        # all?
        x_0, y_0 = centroid(image)
        x_1, y_1 = centroid(image2)
        x_mcp, y_mcp = centroid(sheared1Galaxy)
        x_mcm, y_mcm = centroid(shearedm1Galaxy)
        x_4, y_4 = centroid(image3)
        x_5, y_5 = centroid(image4)

        dx_1 = x_1 - x_0
        dx_m = x_mcp - x_0
        dx_r = x_5 - x_0

        return dx_1, dx_m, dx_r


    
    if shear1_step != 0. and shear2_step == 0.:
        return de1_g1_tru, de1_g1_est, de1_g1_rec
    elif shear2_step != 0. and shear1_step == 0.:
        return de2_g2_tru, de2_g2_est, de2_g2_rec





def main(argv):
    npts = 20
    e_arr =  np.linspace(-0.5, 0.5, npts)
    R_true_arr = e_arr*0.
    R_est_arr = e_arr*0.
    R_rec_arr = e_arr*0.
    shear1_step = 0.0
    shear2_step = 0.01
    e1_intrinsic = 0.
    e2_intrinsic = 0.
    

    
    thing =  metacal_diagnose(e1_intrinsic = e1_intrinsic, e2_intrinsic = e2_intrinsic, shear1_step = shear1_step, shear2_step = shear2_step, doplot=True)

    for i, this_e in zip(xrange(npts), e_arr):
        R_true, R_est, R_rec = metacal_diagnose(e1_intrinsic = 0., e2_intrinsic = this_e,  shear1_step = shear1_step, shear2_step = shear2_step, size = False, do_centroid = False)
        R_true_arr[i] = R_true
        R_est_arr[i] = R_est
        R_rec_arr[i] = R_rec


    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols = 2, figsize = (14,7))
    ax1.plot(e_arr, R_true_arr, label="R_true")
    ax1.plot(e_arr, R_est_arr, label="R_metacal")        
    ax1.plot(e_arr, R_rec_arr, label="R_reconv")
    ax1.axvline(0,color='black',linestyle='--')
    #plt.axhline(2,color='black',linestyle='--')
    ax1.legend(loc='best')

    ax2.plot(e_arr, R_est_arr - R_true_arr, label = "R_est - R_true")
    ax2.plot(e_arr, R_rec_arr - R_true_arr, label = "R_reconv - R_true")
    ax2.axhline(0,color='black',linestyle='--')
    ax2.legend(loc='best')
    plt.show()

    
    
    stop
    
if __name__ == "__main__":
    import pdb, traceback
    try:
        main(sys.argv)
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)

