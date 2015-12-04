import galsim
import math
import numpy as np

def getTargetPSF(psfImage, pixelscale, g1 =0.01, g2 = 0.0, gal_shear=True):
    pixel = galsim.Pixel(pixelscale)

    # Create a GSObj from the psf image.
    l5 = galsim.Lanczos(5, True, 1.0E-4)
    qt = galsim.Quintic()
    psf = galsim.InterpolatedImage(psfImage, k_interpolant=qt)

    # Deconvolve the pixel from the PSF.
    pixInv = galsim.Deconvolve(pixel)
    psfNoPixel = galsim.Convolve([psf , pixInv])

    # Increase the size of the PSF by 2*shear
    psfGrownNoPixel = psfNoPixel.dilate(1 + 2*math.sqrt(g1**2 + g2**2))

    # Convolve the grown psf with the pixel
    psfGrown = galsim.Convolve(psfGrownNoPixel,pixel)

    # I think it's actually the shear of the effective, PSF-convolved PSF that we're sensitive
    # to. So I'm going to shear at this stage if gal_shear is False.
    if not gal_shear:
        psfGrown = psfGrown.shear(g1=g1, g2=g2)

    # Draw to an ImageD object, and then return.
    psfGrownImage = galsim.ImageD(psfImage.bounds)

    psfGrownImage=psfGrown.drawImage(image=psfGrownImage, scale=pixelscale, method='no_pixel')
    return psfGrownImage


def getMetaCalNoiseCorrImage(galaxyImage, psfImage, psfImageTarget, g1=0.0, g2=0.0, variance = None):
    pixel = psfImage.scale
    l5 = galsim.Lanczos(5, True, 1.0E-4)
    qt = galsim.Quintic()
    
    psf = galsim.InterpolatedImage(psfImage, k_interpolant=qt)
    psfTarget = galsim.InterpolatedImage(psfImageTarget, k_interpolant=qt)
    psfInv = galsim.Deconvolve(psf)
    GN = galsim.GaussianNoise(sigma=np.double(np.sqrt(variance)))
    test_im = galsim.Image(512,512,scale=pixel)
    test_im.addNoise(GN)
    CN = galsim.CorrelatedNoise(test_im, scale=pixel)
    print "reported noise before symmetrization is: ",np.sqrt(CN.getVariance())
    CN = CN.convolvedWith(psfInv)
    CN = CN.shear(g1 = g1, g2 = g2)
    CN = CN.convolvedWith(psfTarget)
    print "reported noise after symmetrization is: ",np.sqrt(CN.getVariance())
    noiseCorr = CN.drawImage(image=galaxyImage.copy(),add_to_image=False)
    return noiseCorr, CN

def metaCalibrateReconvolve(galaxyImage, psfImage, psfImageTarget, g1=0.0, g2=0.0, noise_symm = False, variance = None):
    pixel = psfImage.scale
    l5 = galsim.Lanczos(5, True, 1.0E-4)
    qt = galsim.Quintic()
    # Turn the provided image arrays into GSObjects
    # pad factor may be important here (increase to 6?)
    # also, look at k-space interpolant

    galaxy = galsim.InterpolatedImage(galaxyImage, k_interpolant=qt)
    psf = galsim.InterpolatedImage(psfImage, k_interpolant=qt)
    psfTarget = galsim.InterpolatedImage(psfImageTarget, k_interpolant=qt)
    
    # Remove the psf from the galaxy
    psfInv = galsim.Deconvolve(psf)
    galaxy_noPSF = galsim.Convolve(galaxy,psfInv)

    # Apply a shear
    galaxy_noPSF = galaxy_noPSF.shear(g1 = g1, g2 = g2)

    # Reconvolve to the target psf
    galaxy_sheared_reconv = galsim.Convolve([galaxy_noPSF, psfTarget])

    # Draw reconvolved, sheared image to an ImageD object, and return.
    galaxyImageSheared = galsim.ImageD(galaxyImage.bounds)
    galaxyImageSheared = galaxy_sheared_reconv.drawImage(image=galaxyImageSheared,
                                                         method='no_pixel',
                                                         scale=psfImage.scale)

    if noise_symm is True:
        GN = galsim.GaussianNoise(sigma=np.double(np.sqrt(variance)))
        test_im = galsim.Image(512,512,scale=pixel)
        test_im.addNoise(GN)
        CN = galsim.CorrelatedNoise(test_im, scale=pixel)
        print "noise before symmetrization is: ",np.sqrt(CN.getVariance())
        CN = CN.convolvedWith(psfInv)
        CN = CN.shear(g1 = g1, g2 = g2)
        CN = CN.convolvedWith(psfTarget)
        origIm = galaxyImageSheared.copy()
        varCalc = galaxyImageSheared.symmetrizeNoise(CN,order=4)
        #varCalc = CN.whitenImage(galaxyImageSheared)
        print "noise after symmetrization is: ",np.sqrt(varCalc)
    return galaxyImageSheared

def metaCalibrate(galaxyImage, psfImage, g1 = 0.01, g2 = 0.00, gal_shear = True, noise_symm = False, variance = None, targetPSFImage = None):
    """The new gal_shear argument tells metaCalibrate whether to interpret the (g1, g2) args as a
    shear to apply to the *galaxy* (True - which is the default behavior and the only behavior this
    function had before) or to the *PSF* (False - in which case the galaxy is unsheared but the PSF
    is sheared, in addition to being enlarged by the usual amount)."""

    # Routine to drive the metaCalibration procedure.
    pixelscale = psfImage.scale
    # First, work out the target psf, which changes depending on whether we're shearing the galaxy
    # or PSF.  So, propagate that kwarg through.
    if targetPSFImage is None:
        targetPSFImage = getTargetPSF(psfImage, pixelscale, g1 = g1, g2 = g2, gal_shear=gal_shear)
        
    if gal_shear:
        # Then, produce the reconvolved images, with and without shear.
        reconvSheared = metaCalibrateReconvolve(
            galaxyImage, psfImage, targetPSFImage, g1=g1, g2=g2,
            noise_symm = noise_symm, variance = variance)
        reconvUnsheared = metaCalibrateReconvolve(
            galaxyImage, psfImage, targetPSFImage, g1=0.0, g2=0.0,
            noise_symm = noise_symm, variance = variance)
        return reconvSheared, reconvUnsheared, targetPSFImage
    else:
        # We really only have to produce one image since the galaxy isn't sheared.
        reconvUnsheared = \
            metaCalibrateReconvolve(galaxyImage, psfImage, targetPSFImage, g1=0.0,g2=0.0,
            noise_symm = noise_symm, variance = variance)
        return reconvUnsheared, reconvUnsheared, targetPSFImage
