import galsim
import math
import numpy as np

pad_factor = 8
interpolant =  galsim.Quintic()

def getTargetPSF(psfImage, pixelscale, g1 =0.0, g2 = 0.0, gal_shear=True):
    pixel = galsim.Pixel(pixelscale)

    # Create a GSObj from the psf image.
    psf = galsim.InterpolatedImage(psfImage, k_interpolant=interpolant, pad_factor=pad_factor )

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

    return psfGrown


def getMetaCalNoiseCorrImage(galaxyImage, psf, psfTarget, g1=0.0, g2=0.0, variance = None):
    
    psfInv = galsim.Deconvolve(psf)
    CN = galsim.UncorrelatedNoise(variance=variance)
    print "reported noise before metacal is: ",np.sqrt(CN.getVariance())
    CN = CN.convolvedWith(psfInv)
    CN = CN.shear(g1 = g1, g2 = g2)
    CN = CN.convolvedWith(psfTarget)
    print "reported noise after metacal is: ",np.sqrt(CN.getVariance())
    noiseCorr = CN.drawImage(image=galaxyImage.copy(),add_to_image=False)
    return noiseCorr, CN

def metaCalibrateReconvolve(galaxyImage, psf, psfTarget, g1=0.0, g2=0.0, noise_symm = False, variance = None):

    # psf, and psfTarget need to be GSObjects.
    
    # Turn the provided galaxy image into a GSObject
    # pad factor may be important here (increase to 6?)
    # also, look at k-space interpolant
    
    # Remove the psf from the galaxy
    galaxy = galsim.InterpolatedImage(galaxyImage, k_interpolant=interpolant, pad_factor=pad_factor)
    if variance is not None:
        galaxy.noise = galsim.UncorrelatedNoise(variance=variance)
    psfInv = galsim.Deconvolve(psf)
    galaxy_noPSF = galsim.Convolve(galaxy,psfInv)

    # Apply a shear
    galaxy_noPSF = galaxy_noPSF.lens( g1, g2,1.0)

    # Reconvolve to the target psf
    galaxy_sheared_reconv = galsim.Convolve([galaxy_noPSF, psfTarget])
    # Draw reconvolved, sheared image to an ImageD object, and return.
    galaxyImageSheared = galaxy_sheared_reconv.drawImage(image=galaxyImage.copy(),method='no_pixel')
    if noise_symm is True:
        galaxyImageSheared.symmetrizeNoise(galaxy_sheared_reconv.noise, order=4)
        
    return galaxyImageSheared
        
def metaCalibrate(galaxyImage, psfImage, g1 = 0.00, g2 = 0.00, gal_shear = True, noise_symm = False, variance = None,
                  psfObj = None, targetPSFObj = None):
    """The new gal_shear argument tells metaCalibrate whether to interpret the (g1, g2) args as a
    shear to apply to the *galaxy* (True - which is the default behavior and the only behavior this
    function had before) or to the *PSF* (False - in which case the galaxy is unsheared but the PSF
    is sheared, in addition to being enlarged by the usual amount)."""

    # Routine to drive the metaCalibration procedure.
    pixelscale = psfImage.scale
    
    # First, work out the target psf, which changes depending on whether we're shearing the galaxy
    # or PSF.
    if psfObj is None:
        psf = galsim.InterpolatedImage(psfImage, k_interpolant=interpolant, pad_factor=pad_factor)

    if targetPSFObj is None:
        targetPSFObj = getTargetPSF(psfImage, pixelscale, g1 =g1, g2 = g2, gal_shear=gal_shear)

    else:
        psf = psfObj
    
    if gal_shear:
        # Then, produce the reconvolved images, with and without shear.
        reconvSheared = metaCalibrateReconvolve(
            galaxyImage, psf, targetPSFObj, galaxyImage.copy(), g1=g1, g2=g2,
            noise_symm = noise_symm, variance = variance)
        reconvUnsheared = metaCalibrateReconvolve(
            galaxyImage, psf, targetPSFObj, galaxyImage.copy(), g1=0.0, g2=0.0,
            noise_symm = noise_symm, variance = variance)
        return reconvSheared, reconvUnsheared, targetPSFImage
    else:
        # We really only have to produce one image since the galaxy isn't sheared.
        reconvUnsheared = \
            metaCalibrateReconvolve(galaxyImage, psf, targetPSFObj, g1=0.0, g2=0.0,
            noise_symm = noise_symm, variance = variance)
        return reconvUnsheared, reconvUnsheared, targetPSFImage
