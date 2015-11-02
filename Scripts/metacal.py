import galsim
import math

def getTargetPSF(psfImage, pixelscale, g1 =0.01, g2 = 0.0, gal_shear=True):
    pixel = galsim.Pixel(pixelscale)

    # Create a GSObj from the psf image.
    l5 = galsim.Lanczos(5, True, 1.0E-4)
    psf = galsim.InterpolatedImage(psfImage, x_interpolant=l5)

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

def metaCalibrateReconvolve(galaxyImage, psfImage, psfImageTarget, g1=0.0, g2=0.0):
    pixel = psfImage.scale
    l5 = galsim.Lanczos(5, True, 1.0E-4)
    
    # Turn the provided image arrays into GSObjects
    # pad factor may be important here (increase to 6?)
    # also, look at k-space interpolant

    galaxy = galsim.InterpolatedImage(galaxyImage, x_interpolant=l5)
    psf = galsim.InterpolatedImage(psfImage, x_interpolant=l5)
    psfTarget = galsim.InterpolatedImage(psfImageTarget, x_interpolant=l5)
    
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
    # Symmetrize the noise.  We need to check if the info for this is already cached:
    return galaxyImageSheared

def metaCalibrate(galaxyImage, psfImage, g1 = 0.01, g2 = 0.00, gal_shear = True):
    """The new gal_shear argument tells metaCalibrate whether to interpret the (g1, g2) args as a
    shear to apply to the *galaxy* (True - which is the default behavior and the only behavior this
    function had before) or to the *PSF* (False - in which case the galaxy is unsheared but the PSF
    is sheared, in addition to being enlarged by the usual amount)."""

    # Routine to drive the metaCalibration procedure.
    pixelscale = psfImage.scale
    # First, work out the target psf, which changes depending on whether we're shearing the galaxy
    # or PSF.  So, propagate that kwarg through.
    targetPSFImage = getTargetPSF(psfImage, pixelscale, g1 = g1, g2 = g2, gal_shear=gal_shear)
    if gal_shear:
        # Then, produce the reconvolved images, with and without shear.
        reconvSheared = metaCalibrateReconvolve(
            galaxyImage, psfImage, targetPSFImage, g1=g1, g2=g2)
        reconvUnsheared = metaCalibrateReconvolve(
            galaxyImage, psfImage, targetPSFImage, g1=0.0, g2=0.0)
        return reconvSheared, reconvUnsheared, targetPSFImage
    else:
        # We really only have to produce one image since the galaxy isn't sheared.
        reconvUnsheared = \
            metaCalibrateReconvolve(galaxyImage, psfImage, targetPSFImage, g1=0.0,g2=0.0)
        return reconvUnsheared, reconvUnsheared, targetPSFImage
