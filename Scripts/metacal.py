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
    # Convolve one more time with a tiny, tiny gaussian.

    
    # I think it's actually the shear of the effective, PSF-convolved PSF that we're sensitive
    # to. So I'm going to shear at this stage if gal_shear is False.
    if not gal_shear:
        psfGrown = psfGrown.shear(g1=g1, g2=g2)

    return psfGrown


def getMetaCalNoiseCorrImage(galaxyImage, psf, psfTarget, g1=0.0, g2=0.0, variance = None):
    
    psfInv = galsim.Deconvolve(psf)
    CN = galsim.UncorrelatedNoise(variance=variance, rng=galsim.BaseDeviate)
    #print "reported noise before metacal is: ",np.sqrt(CN.getVariance())
    CN = CN.convolvedWith(psfInv)
    CN = CN.shear(g1 = g1, g2 = g2)
    CN = CN.convolvedWith(psfTarget)
    #print "reported noise after metacal is: ",np.sqrt(CN.getVariance())
    noiseCorr = CN.drawImage(image=galaxyImage.copy(),add_to_image=False)
    return CN

def deCorrelateNoiseObject(galaxyImage, psf, psfTarget, g1=0.0, g2=0.0, variance = None, image_size = 256):
    # First, generate a blank image with the right initial noise field.
    noiseImageFull = galsim.Image(image_size,image_size,scale=galaxyImage.scale)
    whiteNoise = galsim.UncorrelatedNoise(variance=variance,rng=galsim.BaseDeviate())
    noiseImageFull.addNoise(whiteNoise)
    # Next, apply metaCal to this image
    noiseImageSheared = metaCalibrateReconvolve(noiseImageFull, psf, psfTarget, g1=g1, g2=g2, noise_symm = False, variance = variance)
    # Find the difference image, representing the correlated noise.
    noiseImageDiff = (  noiseImageSheared - noiseImageFull )
    # create a correlated noise objecting representing this noise.
    deCorrCNObj = galsim.CorrelatedNoise(noiseImageDiff,rng=galsim.BaseDeviate())
    return deCorrCNObj



def metaCalibrateReconvolve(galaxyImage, psf, psfTarget, g1=0.0, g2=0.0, noise_symm = False, variance = None, regularize= True):

    # psf, and psfTarget need to be GSObjects.
    # psf and psfTarget should both contain the pixel.
    # Turn the provided galaxy image into a GSObject
    
    # Remove the psf from the galaxy
    
    galaxy = galsim.InterpolatedImage(galaxyImage, k_interpolant=interpolant, pad_factor=pad_factor)
    
    if variance is not None:
        galaxy.noise = galsim.UncorrelatedNoise(variance=variance)
    psfInv = galsim.Deconvolve(psf)
    galaxy_noPSF = galsim.Convolve([galaxy,psfInv])

    # Apply a shear
    galaxy_noPSF = galaxy_noPSF.lens( g1, g2,1.0)

    # Reconvolve to the target psf
    galaxy_sheared_reconv = galsim.Convolve([galaxy_noPSF, psfTarget])

    
    # Draw reconvolved, sheared image to an ImageD object.
    galaxyImageSheared = galaxy_sheared_reconv.drawImage(image=galaxyImage.copy(),method='no_pixel')
    
    if regularize is True:
        # apply Bernstein's k-sigma filter.
        imFFT = np.fft.fft2(galaxyImageSheared.array)
        ky = np.fft.fftfreq(galaxyImageSheared.array.shape[0])
        kyy = np.outer(ky,np.ones(galaxyImageSheared.array.shape[1]))
        kx =  np.fft.fftfreq(galaxyImageSheared.array.shape[1])
        kxx = np.outer(np.ones(galaxyImageSheared.array.shape[0]),kx)
        kk = np.sqrt(kxx**2 + kyy**2)
        scale = (2./galaxyImage.scale)
        Nfilt = 4
        outer = kk >= np.sqrt(2*Nfilt)/scale
        W = (1 - (kk*scale)**2/(2*Nfilt))**Nfilt
        W[outer] =0.
        imFFT = imFFT * W
        arrFilt = np.ascontiguousarray(np.real(np.fft.ifft2(imFFT)))
        imFilt = galsim.Image(arrFilt,scale=galaxyImageSheared.scale)
        #imFilt.addNoise(galsim.UncorrelatedNoise(variance=variance))
        #imFilt.noise = galsim.UncorrelatedNoise(variance=variance)
        return imFilt
        
    
    if noise_symm is True:
        #deCorrNoiseObj = deCorrelateNoiseObject(galaxyImage, psf, psfTarget, g1=g1, g2=g2,variance = variance)
        #galaxyImageSheared.addNoise(deCorrNoiseObj)
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
