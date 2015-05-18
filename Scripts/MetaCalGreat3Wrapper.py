#!/usr/bin/env python
#
# This code is copied heavily from an example script developed by
# Rachel Mandelbaum and the Great3 team.
#
# The purpose of this script is to run Eric Huff's metaCalibration
# routine on Great3 galaxy simulations.

import sys
import time
import os
import optparse
import numpy
import socket

try:
    import astropy.io.fits as pyfits
except:
    import pyfits

import galsim

default_shear_kwds = {"strict": False}
verbose = False


def log(msg):
    if verbose:
        print msg

def readData(sim_dir, subfield, coadd, variable_psf_dir=""):
    """Subroutine to read in galaxy and star field images and the galaxy catalog for the specified
    simulation.

    Arguments:

      sim_dir ----------- simulation directory, containing the GREAT3 images and catalogs for a
                          single branch

      subfield ---------- subfield to be processed, as an integer

      coadd ------------- Are we processing the outputs of the coaddition script,
                          coadd_multiepoch.py? If so, set this to True.

      variable_psf_dir--- Directory in which to find PSF model outputs from psf_models.py, if this
                          is a variable PSF branch.  Default value of "" indicates that this is not
                          a variable_psf branch.

    """
    # Construct filename for galaxy image, and read in file.
    if not coadd:
        infile = os.path.join(sim_dir, 'image-%03d-0.fits'%subfield)
    else:
        infile = os.path.join(sim_dir, 'coadd_image-%03d.fits'%subfield)
    try:
        gal_im = galsim.fits.read(infile)
    except:
        raise RuntimeError("Could not read in file %s."%infile)
    log("... Read galaxy image from file "+infile)

    # Construct filename for starfield image, and read in file.  There are three options: a standard
    # control/real_galaxy star field; a variable_psf grid of PSF models, one per galaxy; and a coadd
    # starfield.
    if not coadd and variable_psf_dir=="":
        # This is a standard star field.
        infile = os.path.join(sim_dir, 'starfield_image-%03d-0.fits'%subfield)
    elif variable_psf_dir != "":
        # This is a grid of PSF models for a variable PSF branch.
        infile = os.path.join(variable_psf_dir, 'psf_models-%03d.fits'%subfield)
    else:
        # This is a coadd.
        infile = os.path.join(sim_dir, 'coadd_starfield_image-%03d.fits'%subfield)
    try:
        starfield_im = galsim.fits.read(infile)
    except:
        raise RuntimeError("Could not read in file %s."%infile)
    log("... Read starfield image from file "+infile)

    # Construct filename for galaxy catalog, and read in file.
    infile = os.path.join(sim_dir, 'galaxy_catalog-%03d.fits'%subfield)
    try:
        gal_catalog = pyfits.getdata(infile)
    except:
        raise RuntimeError("Could not read in file %s."%infile)
    log("... Read galaxy catalog from file "+infile)

    return gal_im, starfield_im, gal_catalog

def extractPSF(starfield_im):
    """Subroutine to extract a single PSF image from a 3x3 starfield.

    This routine assumes we are in one of the constant PSF branches, such that the starfield is
    simply a 3x3 grid of stars, for which we wish to extract the lower-left corner (containing a
    star that is perfectly centered in the postage stamp).

    Arguments:

      starfield_im ----- The full starfield image from which we wish to extract the PSF for this
                         galaxy image.

    """
    # Figure out the size of the images based on knowing that the starfield image is a 3x3 grid of
    # stars.
    shape = starfield_im.array.shape
    if shape[0] != shape[1]:
        raise RuntimeError("This starfield image is not square!")
    if shape[0] % 3 != 0:
        raise RuntimeError("Starfield image size is not a multiple of 3!")
    ps_size = shape[0] / 3

    # Cut out the lower-left postage stamp.
    psf_im = starfield_im[galsim.BoundsI(0,ps_size-1,0,ps_size-1)]
    psf_im.wcs = galsim.PixelScale(0.2)

    
    return psf_im


def estimateVariance(gal_im):
    """Subroutine to do a fairly naive estimation of the sky variance, using edge pixels.

    This routine uses the fact that the sky variance is the same across the entire large image, so
    we can use a set of pixels at the edge (with minimal contamination from galaxy fluxes) in order
    to estimate the variance.

    Arguments:
      gal_im -------- The full galaxy image to use for estimating the sky variance.

    """
    # Figure out the size of the images
    shape = gal_im.array.shape
    if shape[0] != shape[1]:
        raise RuntimeError("This image is not square!")

    # Choose the 8 rows/columns and make a single array with sky pixels to use
    sky_vec = numpy.concatenate(
        (gal_im.array[0:2, 2:shape[0]-2].flatten(),
         gal_im.array[shape[0]-2:shape[0], 2:shape[0]-2].flatten(),
         gal_im.array[2:shape[0]-2, 0:2].flatten(),
         gal_im.array[2:shape[0]-2, shape[0]-2:shape[0]].flatten()
         ))

    # Estimate and return the variance
    return numpy.var(sky_vec)

def getPS(record, gal_im, ps_size, starfield_image=None):
    """Routine to pull out a galaxy postage stamp based on a record from a catalog.

    Arguments:

      record ---------- A single record from a GREAT3 galaxy catalog for the chosen subfield.

      gal_im ---------- The full galaxy image for that subfield.

      ps_size --------- Total (linear) size of a single postage stamp.

      starfield_image - Grid of PSF models for the galaxies, if this is a variable PSF branch.  In
                        that case, the routine returns not just the galaxy postage stamp, but also
                        the PSF postage stamp.
    """
    # Figure out the galaxy postage stamp bounds based on the record information.
    radius = ps_size / 2
    bounds = galsim.BoundsI(int(numpy.ceil(record['x']) - radius) + 1,
                            int(numpy.ceil(record['x']) + radius),
                            int(numpy.ceil(record['y']) - radius) + 1,
                            int(numpy.ceil(record['y']) + radius))

    # Pull out and return the postage stamp.
    subimage = gal_im[bounds]

    if starfield_image is None:
        return subimage
    else:
        # Then work on the PSF image, if this is a variable PSF branch.
        # This is a general way to find the dimensions of a PSF postage stamp, just in case users
        # use a non-default option for psf_models.py.
        psf_dim = (starfield_image.xmax + 1 - starfield_image.xmin) * ps_size / \
            (gal_im.xmax + 1 - gal_im.xmin)
        # Now find the proper indices
        x_index = (record['x'] + 1 - radius) / ps_size
        y_index = (record['y'] + 1 - radius) / ps_size
        # Take the subimage.
        bounds = galsim.BoundsI(int(x_index*psf_dim), int((x_index+1)*psf_dim)-1,
                                int(y_index*psf_dim), int((y_index+1)*psf_dim)-1)
        psf_subimage = starfield_image[bounds]
        return subimage, psf_subimage

def checkFailures(shear_results, responsivity_stat):
    """Routine to check for shape measurement failures which should be flagged as such.

    Arguments:

      shear_results --- a list of galsim.hsm.ShapeData objects that comes from measuring shapes of
                        all galaxies in the image.

    """
    n_gal = len(shear_results)

    # Define output structure: a boolean numpy array
    use_shape = numpy.ones(n_gal).astype(bool)

    # Compare resolution factor with galsim.hsm.HSMParams.failed_moments or look for other obvious
    # oddities in shears, quoted errors, etc.
    hsmparams = galsim.hsm.HSMParams()
    for index in range(n_gal):
        test_e = numpy.sqrt(
            shear_results[index].corrected_e1**2 + shear_results[index].corrected_e2**2
            )
        if shear_results[index].resolution_factor == hsmparams.failed_moments or \
                shear_results[index].corrected_shape_err < 0 or test_e > 4. or \
                shear_results[index].corrected_shape_err > 0.5 or \
                responsivity_stat[index]==0:
            use_shape[index] = False
    return use_shape



def writeOutput(output_dir, output_prefix, subfield, output_type, catalog, clobber = True,
                comment_pref = '#'):
    """Routine for writing outputs to file in some specified format, either fits or ascii.

    Arguments:

      output_dir ------ Output directory.

      output_prefix --- Prefix for output files.

      subfield -------- Subfield that was processed.

      output_type ----- Type of file to output, either FITS or ascii.

      catalog --------- Catalog to write to file.

      clobber --------- Overwrite catalog if it already exists?  Default: true.

      comment_pref ---- String to use to denote comments in ascii outputs.  Default: '#'
    """
    if output_type == "fits":
        output_file = '%s-%03d.fits'%(output_prefix, subfield)
    else:
        output_file = '%s-%03d.dat'%(output_prefix, subfield)
    output_file = os.path.join(output_dir, output_file)
    if not clobber and os.path.exists(output_file):
        raise RuntimeError("Error: file %s already exists!" % output_file)

    if output_type == "fits":
        pyfits.writeto(output_file, catalog, clobber = clobber)
        log("Wrote output catalog to file %s."%output_file)
    else:
        import tempfile
        import shutil
        # First print lines with column names.  This goes into a separate file for now.
        f1, tmp_1 = tempfile.mkstemp(dir=output_dir)
        with open(tmp_1, 'w') as f:
            name_list = catalog.dtype.names
            for name in name_list:
                f.write(comment_pref+" "+name+"\n")

        # Then save catalog itself.
        f2, tmp_2 = tempfile.mkstemp(dir=output_dir)
        numpy.savetxt(tmp_2, catalog, ["%d","%.8f","%.8f","%.8f"])

        # Finally, concatenate, and remove tmp files
        with open(output_file, 'wb') as destination:
            shutil.copyfileobj(open(tmp_1, 'rb'), destination)
            shutil.copyfileobj(open(tmp_2, 'rb'), destination)
        # Need to close the tempfiles opened by mkstemp.  cf:
        # http://stackoverflow.com/questions/9944135/how-do-i-close-the-files-from-tempfile-mkstemp
        log("Wrote output catalog to file "+output_file)
        os.close(f1)
        os.close(f2)
        os.remove(tmp_1)



def getTargetPSF(psfImage, pixelscale, g1 =0.01, g2 = 0.0, gal_shear=True):
    pixel = galsim.Pixel(pixelscale)

    # Create a GSObj from the psf image.
    l5 = galsim.Lanczos(5, True, 1.0E-4)
    l52d = galsim.InterpolantXY(l5)
    psf = galsim.InterpolatedImage(psfImage, x_interpolant = l52d)

    # Deconvolve the pixel from the PSF.
    pixInv = galsim.Deconvolve(pixel)
    psfNoPixel = galsim.Convolve([psf , pixInv])

    # Increase the size of the PSF by 2*shear
    psfGrownNoPixel = psfNoPixel.dilate(1 + 2*numpy.max([g1,g2]))

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


def metaCalibrateReconvolve(galaxyImage, psfImage, psfImageTarget, g1 = 0.01, g2 = 0.0,variance=1.):
    
    pixel = psfImage.scale
    l5 = galsim.Lanczos(5, True, 1.0E-4)
    l52d = galsim.InterpolantXY(l5)
    # Turn the provided image arrays into GSObjects
    galaxy = galsim.InterpolatedImage(galaxyImage, x_interpolant = l52d)
    psf = galsim.InterpolatedImage(psfImage, x_interpolant = l52d)
    psfTarget = galsim.InterpolatedImage(psfImageTarget, x_interpolant = l52d)
    
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


    # Symmetrize the noise.
    # For this we need to know something about the noise field. Let this be represented by the noise object CN.
    # Initialize as uncorrelated noise with fixed (known) variance.
    GN = galsim.GaussianNoise(sigma=numpy.double(numpy.sqrt(variance)))
    test_im = galsim.Image(512,512,scale=pixel)
    test_im.addNoise(GN)
    CN = galsim.CorrelatedNoise(test_im,scale=pixel)
    # Now apply the same set of operations to this...
    CN = CN.convolvedWith(psfInv)
    CN = CN.shear(g1 = g1, g2 = g2)
    CN = CN.convolvedWith(psfTarget)
    varCalc = galaxyImageSheared.symmetrizeNoise(CN,order=4)
    varEst = estimateVariance(galaxyImageSheared)
    print 'Estimated, Calculated noise after reconvolution:', varEst, varCalc
    print 'Input noise estimate:', variance
    return galaxyImageSheared


def metaCalibrate(galaxyImage, psfImage, g1 = 0.01, g2 = 0.00, gal_shear = True, variance = 0.01):
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
        reconvSheared   = metaCalibrateReconvolve(galaxyImage, psfImage, targetPSFImage, g1 = g1, g2 = g2,variance = variance)
        reconvUnsheared = metaCalibrateReconvolve(galaxyImage, psfImage, targetPSFImage, g1 = 0.0, g2 = 0.0, variance = variance)
        return reconvSheared, reconvUnsheared, targetPSFImage
    else:
        # We really only have to produce one image since the galaxy isn't sheared.
        reconvUnsheared = \
            metaCalibrateReconvolve(galaxyImage, psfImage, targetPSFImage, g1=0.0, g2=0.0,variance=variance)
        return reconvUnsheared, reconvUnsheared, targetPSFImage


def getRMSEllip(shear_results=None, use_shape=None, weight=None, e1=None, e2=None, sigma_e=None):
    """Routine for estimating the RMS ellipticity from a catalog.

    The routine works in one of two ways:

    In the first approach, if we have not done any shape calculations before, we can pass in an
    array of galsim.hsm.ShapeData structures for every object in the image (`shear_results`), a
    boolean NumPy array indicating which ones are useful, and (optionally) a set of weights.  The
    routine will then return the RMS ellipticity (optionally weighted) as well as arrays with e1,
    e2, sigma_e for each object used.

    In the second approach, if we've already gotten e1, e2, sigma_e before, and we just want to
    recalculate RMS ellipticity for some reason (e.g., with weights), then we can pass in e1, e2,
    sigma_e along with an array of weights of the same length, and calculate the weighted RMS
    ellipticity, which is the sole quantity that will be returned.
    """
    mode = None
    if e1 is None and e2 is None and sigma_e is None:
        if shear_results is None or use_shape is None:
            raise RuntimeError("Missing information: need ShapeData and usage flags!")
        else:
            mode = 1

    if shear_results is None and use_shape is None:
        if e1 is None or e2 is None or sigma_e is None:
            raise RuntimeError("Missing information: need e1, e2, sigma_e!")
        else:
            mode = 2

    if mode is None:
        raise RuntimeError("getRMSEllip called without an obvious working mode!")

    if mode == 1:
        n_gal = len(shear_results)
        e1 = []
        e2 = []
        sigma_e = []
        for index in range(n_gal):
            if use_shape[index]:
                e1.append(shear_results[index].corrected_e1)
                e2.append(shear_results[index].corrected_e2)
                # There's a factor of ~2 here, because we want sigma_e and it returns sigma_g.
                # Actually, the exact factor is 2/(1+2*sigma_g^2) according to Bernstein & Jarvis
                # (2002), but the sigma_g values are usually <~0.15 so the denominator is quite
                # close to 1 and therefore ignored for the sake of simplicity.
                sigma_e.append(2.*shear_results[index].corrected_shape_err)
        e1 = numpy.array(e1)
        e2 = numpy.array(e2)
        sigma_e = numpy.array(sigma_e)

    n_shape = len(e1)
    if weight is None:
        weight = numpy.ones(n_shape)
    e_rms_per_component = numpy.sqrt(
        numpy.sum(0.5*(e1**2 + e2**2 - 2*sigma_e**2)*weight) / numpy.sum(weight)
        )
    if mode == 1:
        return e_rms_per_component, e1, e2, sigma_e
    else:
        return e_rms_per_component


def EstimateAllShearsStar(args):
    # This is a convenience function for multiprocessing.
    # Python's pool.map seems only to want to deal with functions of a single argument.
    return EstimateAllShears(*args)

def EstimateAllShears(subfield, sim_dir, output_dir, output_prefix="output_catalog", output_type="fits",
                      clobber=True, sn_weight=False, calib_factor=0.98, coadd=False,
                      variable_psf_dir=""):
    """Main driver for all routines in this file, and the implementation of most of the command-line
    interface.

    This routine has three distinct steps:

      1) Read required inputs from file.

      2) Loop over objects in the catalog and estimate a shear for each.

      3) Collect outputs into the proper format, and write to disk.

    Required arguments:

      subfield ----- subfield to be processed, as an integer

      sim_dir ------ simulation directory, containing the GREAT3 images and catalogs for a single
                     branch

      output_dir --- directory in which outputs should be placed

    Optional arguments:

      output_prefix --- Prefix for output catalog; the subfield (as a 3-digit number) will be
                        appended

      output_type ----- Type for output catalogs: fits (default) or ascii.

      clobber --------- Overwrite pre-existing output files?  Default: true.

      sn_weight ------- Apply S/N-dependent weighting to each object?  Default: false.  If false, all
                        objects get an equal weight.

      calib_factor ---- Multiplicative calibration factor to apply to all shears.  Default value of
                        0.98 is based on estimates from previously published work; see full
                        docstring in simple_shear.py for details.

      coadd ----------- Are we processing the outputs of the coaddition script, coadd_multiepoch.py?
                        If so, set this to True.  Default: false.

      variable_psf_dir- Directory in which to find PSF model outputs from psf_models.py, if this is
                        a variable PSF branch.  Default value of "" indicates that this is not a
                        variable_psf branch.

    """
    if coadd is True and variable_psf_dir!="":
        raise NotImplementedError("Script is not set up to process full experiment.")
    if variable_psf_dir!="":
        raise NotImplementedError("Self calibration is only set up for constant PSF.")

    t1 = time.time()
    log("Preparing to read inputs: galaxy and starfield images, and galaxy catalog.")
    # First read inputs.
    gal_im, starfield_im, gal_catalog = readData(sim_dir, subfield, coadd=coadd,
                                                 variable_psf_dir=variable_psf_dir)

    #gal_catalog=gal_catalog[0:2]
    # In order for the deconvolution to work, we really need the wcs to be a simple pixel scale. The units are totally arbitrary.
    gal_im.wcs = galsim.PixelScale(0.2)
    starfield_im.wcs = galsim.PixelScale(0.2)
    pix = gal_im.scale
    
    guess_sig = 3.0 # the default guess for PSF sigma
    if variable_psf_dir=="":
        # Since this is a constant PSF branch, make a PSF image from the lower-left (centered) star
        # in the starfield.  We don't have to do this if we already have the output of the
        # coaddition script, which is a single star.
        if coadd:
            psf_im = starfield_im
        else:
            psf_im = extractPSF(starfield_im)
        # Very rarely, the EstimateShear routine can fail for space PSFs that are very aberrated.  A
        # simple fix for this is to change the initial guess for PSF size.  However, that leads to
        # slower convergence for the 99.9% of subfields that don't have this problem, so we don't
        # want to do it all the time.  Instead, we check once for failure of adaptive moments for
        # the PSF, and if it fails, then we adopt a smaller initial guess.
        try:
            galsim.hsm.FindAdaptiveMom(psf_im)
        except:
            guess_sig = 2.0
            try:
                galsim.hsm.FindAdaptiveMom(psf_im, guess_sig=guess_sig)
            except:
                raise RuntimeError("Cannot find a value of PSF size that works for this PSF!")

    # Get number of entries
    n_gal = len(gal_catalog)
    log("Preparing to process %d galaxies"%n_gal)

    # We need to give the shear estimation routines a rough idea of the sky variance, so that it can
    # estimate uncertainty in per-object shears.  For this purpose, we compute the variance from a
    # set of pixels around the outer edge of the image.
    sky_var = estimateVariance(gal_im)
    log("Estimated sky variance: %f"%sky_var)


    # MetaCalibration takes a galaxy image and a psf image, and
    # returns a model of what that image would have looked like when
    # sheared.
    shear_results = []
    responsivity_stat = []
    responsivity_1 = []
    responsivity_2 = []
    additive_1 = []
    additive_2 = []
    anisotropy_1 = []
    anisotropy_2 = []
    psf_e1 = []
    psf_e2 = []
    ps_size = gal_catalog['x'][1] - gal_catalog['x'][0] # size of a postage stamp defined by
                                                        # centroid differences
    
    # Loop over objects in catalog.
    log("Beginning loop over galaxies: get postage stamp and estimate distortion...")
    t2 = time.time()
    ii_alive = 0
    for record in gal_catalog:
        ii_alive = ii_alive+1
        if ii_alive % 100 == 0:
            print "subfield: ",subfield,", galaxy: ",ii_alive
        if variable_psf_dir=="":
            gal_ps = getPS(record, gal_im, ps_size)
        else:
            starfield_im.setOrigin(0,0)
            gal_ps, psf_im = getPS(record, gal_im, ps_size, starfield_image=starfield_im)

        # Estimate the shear, requiring a silent failure if something goes wrong.  (That flag is in
        # `default_shear_kwds`)
        
        # We need to infer the background noise somehow, for use in the metaCalibration noise symmetrization.
        variance = estimateVariance(gal_ps)
        
        # Here are the bits that are needed for a calibration bias (multiplicative) correction.
        
        sheared1Galaxy, unsheared1Galaxy, reconv1PSF = metaCalibrate(gal_ps, psf_im, g1 = 0.01, g2 = 0.00, variance = variance)
        shearedm1Galaxy, unshearedm1Galaxy, reconvm1PSF = metaCalibrate(gal_ps, psf_im, g1 = -0.01, g2 = 0.00, variance = variance)
        sheared2Galaxy, unsheared2Galaxy, reconv2PSF = metaCalibrate(gal_ps, psf_im, g1 = 0.00, g2 = 0.01, variance = variance)
        shearedm2Galaxy, unshearedm2Galaxy, reconvm2PSF = metaCalibrate(gal_ps, psf_im, g1 = 0.00, g2 = -0.01, variance = variance)
        
        # These new bits make some images that we need for a PSF anisotropy correction.
        unsheared1PGalaxy, _, reconv1PPSF = metaCalibrate(gal_ps, psf_im, g1=0.01, g2=0.0, gal_shear=False, variance = variance)
        unshearedm1PGalaxy, _, reconvm1PPSF = metaCalibrate(gal_ps, psf_im, g1=-0.01, g2=0.0, gal_shear=False, variance = variance)
        unsheared2PGalaxy, _, reconv2PPSF = metaCalibrate(gal_ps, psf_im, g1=0.0, g2=0.01, gal_shear=False, variance = variance)
        unshearedm2PGalaxy, _, reconvm2PPSF = metaCalibrate(gal_ps, psf_im, g1=0.0, g2=-0.01, gal_shear=False, variance = variance)

        #gal_ps.write("./MCImages/originalImage.%d.%d.fits" % (subfield, ii_alive))
        #sheared1Galaxy.write("./MCImages/shearedImage1.%d.%d.fits" % (subfield, ii_alive))
        #unsheared1Galaxy.write("./MCImages/unShearedImage1.%d.%d.fits" % (subfield, ii_alive))
        #reconv1PSF.write("./MCImages/reconvPSF1.%d.%d.fits" % (subfield, ii_alive))
        

        res = galsim.hsm.EstimateShear(unsheared1Galaxy, reconv1PSF, sky_var=float(sky_var),
                                       guess_sig_PSF = guess_sig, shear_est="REGAUSS",  **default_shear_kwds)

        shear_results.append(res)
        try:
            sky_var = float(sky_var)
            # Things needed for multiplicative bias etc.
            res_g1 = galsim.hsm.EstimateShear(sheared1Galaxy, reconv1PSF, sky_var=sky_var,
                                              guess_sig_PSF = guess_sig, shear_est="REGAUSS")

            res_g2 = galsim.hsm.EstimateShear(sheared2Galaxy, reconv2PSF, sky_var=sky_var,
                                              guess_sig_PSF = guess_sig, shear_est="REGAUSS")

            res_mg1 = galsim.hsm.EstimateShear(shearedm1Galaxy, reconvm1PSF, sky_var=sky_var,
                                               guess_sig_PSF = guess_sig, shear_est="REGAUSS")

            res_mg2 = galsim.hsm.EstimateShear(shearedm2Galaxy, reconvm2PSF, sky_var=sky_var,
                                               guess_sig_PSF = guess_sig, shear_est="REGAUSS")
            # Things needed for additive bias
            res_g1p = galsim.hsm.EstimateShear(unsheared1PGalaxy, reconv1PPSF, sky_var=sky_var,
                                               guess_sig_PSF = guess_sig, shear_est="REGAUSS")
            res_g2p = galsim.hsm.EstimateShear(unsheared2PGalaxy, reconv2PPSF, sky_var=sky_var,
                                               guess_sig_PSF = guess_sig, shear_est="REGAUSS")
            res_mg1p = galsim.hsm.EstimateShear(unshearedm1PGalaxy, reconvm1PPSF, sky_var=sky_var,
                                               guess_sig_PSF = guess_sig, shear_est="REGAUSS")
            res_mg2p = galsim.hsm.EstimateShear(unshearedm2PGalaxy, reconvm2PPSF, sky_var=sky_var,
                                               guess_sig_PSF = guess_sig, shear_est="REGAUSS")
            psf_mom = galsim.hsm.FindAdaptiveMom(reconv1PSF)
            
            # Get most of what we need to make a derivative, i.e., (distortion with +g applied) -
            # (distortion with -g applied).  Then divide by 2 since we're doing a 2-sided
            # derivative.  Later, we'll sum these up with weights, and divide by the applied shear.
            # This is for multiplicative bias:
            de1_g1 = 0.5*(res_g1.corrected_e1 - res_mg1.corrected_e1)/0.01
            de2_g2 = 0.5*(res_g2.corrected_e2 - res_mg2.corrected_e2)/0.01
            # Basic systematics correction for regauss oddities:
            c1 = 0.5 * (res_g1.corrected_e1 + res_mg1.corrected_e1) - res.corrected_e1
            c2 = 0.5 * (res_g2.corrected_e2 + res_mg2.corrected_e2) - res.corrected_e2
#            if numpy.random.randomu() <= 0.1:
#                print "measured responsivity: ",de1_g1
            # This is the new stuff for additive PSF anisotropy correction:
            de1_dpg1 = 0.5*(res_g1p.corrected_e1 - res_mg1p.corrected_e1)/0.01
            de2_dpg2 = 0.5*(res_g2p.corrected_e2 - res_mg2p.corrected_e2)/0.01
            responsivity_stat.append(1)
            responsivity_1.append(de1_g1)
            responsivity_2.append(de2_g2)
            additive_1.append(c1)
            additive_2.append(c2)
            anisotropy_1.append(de1_dpg1)
            anisotropy_2.append(de2_dpg2)
            psf_e1.append(psf_mom.observed_shape.e1)
            psf_e2.append(psf_mom.observed_shape.e2)
        except:
            responsivity_stat.append(0)
            responsivity_1.append(-10.)
            responsivity_2.append(-10.)            
            additive_1.append(-10)
            additive_2.append(-10)
            anisotropy_1.append(-10)
            anisotropy_2.append(-10)
            psf_e1.append(-10)
            psf_e2.append(-10)

    dt = time.time() - t2
    log("...Time per object=%f s, total time for loop=%f s"%(dt/n_gal,dt))

    # First figure out which objects have failures.
    use_shape = checkFailures(shear_results, responsivity_stat)
    n_success = numpy.round(use_shape.sum())
    log("Number with successful measurements: %d, or %f percent"%
        (n_success,100*float(n_success)/n_gal))

    # Estimate RMS ellipticity.
    e_rms_per_component, e1, e2, sigma_e = getRMSEllip(shear_results=shear_results,
                                                       use_shape=use_shape, weight=None)
    log("Estimated RMS ellipticity: %f"%e_rms_per_component)

    # Use that to define per-object weights.
    if sn_weight:
        use_weight = 1./(e_rms_per_component**2 + sigma_e**2)

    # Now get a weighted responsivity:
    tot_de1_dg1 = 0.
    tot_de2_dg2 = 0.
    tot_c1 = 0.
    tot_c2 = 0.

    use_tot_de1_dg1 = []
    use_tot_de2_dg2 = []
    use_tot_c1 = []
    use_tot_c2 = []
    tot_wt = 0.0
    i_use = 0
    for i_gal in range(n_gal):
        if sn_weight:
            # Check if it's something we're supposed to use:
            if use_shape[i_gal]:
                # Accumulate sums
                tot_de1_dg1 += use_weight[i_use]*responsivity_1[i_gal]
                tot_de2_dg2 += use_weight[i_use]*responsivity_2[i_gal]
                tot_c1 += use_weight[i_use]* additive_1[i_gal]
                tot_c2 += use_weight[i_use]* additive_2[i_gal]
                tot_wt += use_weight[i_use]
                use_tot_de1_dg1.append(responsivity_1[i_gal])
                use_tot_de2_dg2.append(responsivity_2[i_gal])
                i_use += 1
        else:
            if use_shape[i_gal]:
                tot_de1_dg1 += responsivity_1[i_gal]
                tot_de2_dg2 += responsivity_2[i_gal]
                tot_c1 += additive_1[i_gal]
                tot_c2 += additive_2[i_gal]
                tot_wt += 1.0
                i_use += 1
    tot_de1_dg1 /= (tot_wt)
    tot_de2_dg2 /= (tot_wt)
    tot_c1 /= tot_wt
    tot_c2 /= tot_wt

    log("Responsivities: %f %f"%(tot_de1_dg1,tot_de2_dg2))
    log("Additive bias: %f %f" %(tot_c1, tot_c2))

    # Save some per-object stats, for testing purposes
    schema_tmp = [("d1",float),("d2",float)]
    catalog = numpy.zeros(len(use_tot_de1_dg1), dtype=numpy.dtype(schema_tmp))
    catalog["d1"] = use_tot_de1_dg1
    catalog["d2"] = use_tot_de2_dg2
    writeOutput(output_dir, 'testing', subfield, output_type, catalog, clobber = clobber)
    
    # Now put outputs into the format that we want.  For default values of g1 and g2, we put 100,
    # to indicate failure.  Then we will change them to some more sensible value if a shape was
    # measurable for that object.
    schema = [("id", int), ("g1", float), ("g2", float), ("R1",float),("R2",float),("weight", float),
              ("c1",float),("c2",float),("a1",float),("a2",float),("psf_e1",float),("psf_e2",float)]
    catalog = numpy.zeros(n_gal, dtype=numpy.dtype(schema))
    catalog["id"] = gal_catalog['ID']
    g1 = numpy.zeros(n_gal).astype(numpy.float64) + 100.
    g2 = numpy.zeros(n_gal).astype(numpy.float64) + 100.
    weight = numpy.zeros(n_gal).astype(numpy.float64) + 1.

    index = 0
    use_index = 0

    for index in range(n_gal):
        g1[index] = shear_results[index].corrected_e1
        g2[index] = shear_results[index].corrected_e2
        if sn_weight:
            weight[index] = use_weight[use_index]
        if not use_shape[index]:
            weight[index] = 0.
            use_index += 1
    catalog["g1"] = g1
    catalog["g2"] = g2
    catalog["weight"] = weight
    catalog["R1"] = responsivity_1
    catalog["R2"] = responsivity_2
    catalog["c1"] = additive_1
    catalog["c2"] = additive_2
    catalog["a1"] = anisotropy_1
    catalog["a2"] = anisotropy_2
    catalog["psf_e1"] = psf_e1
    catalog["psf_e2"] = psf_e2

    # Write to file.
    writeOutput(output_dir, output_prefix, subfield, output_type, catalog, clobber = clobber)
    log("Total time processing this subfield: %f s."%(time.time()-t1))

def EstimateAllShearsStar(args):
    # This is a convenience function for multiprocessing.
    # Python's pool.map seems only to want to deal with functions of a single argument.
    return EstimateAllShears(*args)

def main(argv):
    usage = "usage: %prog [options] SUBFIELD SIM_DIR WORK_DIR"
    description = """Estimate shears for all galaxies in the given subfield, applying all necessary responsivity and calibration factors.  SUBFIELD is the number of the subfield to be processed. SIM_DIR is the directory containing GREAT3 images and catalogs for the branch of interest. WORK_DIR is the directory where output files should be placed.  It will be created if it does not exist."""
    parser = optparse.OptionParser(usage=usage, description=description)
    parser.add_option("--output_prefix", dest="output_prefix", type=str, default="output_catalog",
                      help="Prefix for output file")
    parser.add_option("--output_type", dest="output_type", type=str, default="fits",
                      help="Type of output catalog: fits or ascii")
    parser.add_option("--no_clobber", action="store_true",
                      help="Do not clobber pre-existing output files")
    parser.add_option("--no_weight", action="store_true",
                      help="Do not apply S/N-dependent weighting; weight all galaxies equally")
    parser.add_option("--calib_factor", dest="calib_factor", type=float, default=0.98,
                      help="Multiplicative calibration factor to apply to all shears, default 0.98")
    parser.add_option("--coadd", dest="coadd", action='store_true', default=False,
                      help="Use to indicate that we are processing coadd_multiepoch.py outputs")
    parser.add_option("--variable_psf_dir", dest="variable_psf_dir", type=str, default="",
                      help="Directory in which to find variable PSF models; only use this option for a variable PSF branch!")
    parser.add_option("--quiet", dest="quiet", action='store_true', default=False,
                      help="Don't print progress statements")
    opts, args = parser.parse_args()
    try:
        subfield, sim_dir, output_dir = args
    except ValueError:
        parser.error("exactly three positional arguments are required")
    if not os.path.isdir(sim_dir):
        parser.error("input directory %s does not exist or is not a directory" % sim_dir)
    try:
        subfield = int(subfield)
    except TypeError:
        parser.error("subfield argument '%s' is not an integer" % subfield)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    if opts.output_type is not None:
        if opts.output_type not in ("fits", "ascii"):
            parser.error("output_type '%s' must be one of 'fits' or 'ascii'" % opts.output_type)
    if opts.no_clobber:
        clobber = False
    else:
        clobber = True
    if opts.no_weight:
        sn_weight = False
    else:
        sn_weight = True
    global verbose
    if opts.quiet:
        verbose = False
    else:
        verbose = True

    if 'coma' in socket.gethostname():
        # Just use one CPU on the coma cluster, since we've already made this whole thing
        # embarrassingly parallel by farming out each subfield to a single CPU.
        EstimateAllShears(
            subfield, sim_dir, output_dir,
            output_prefix=opts.output_prefix,
            output_type=opts.output_type,
            clobber=clobber,
            sn_weight=sn_weight,
            calib_factor=opts.calib_factor,
            coadd=opts.coadd,
            variable_psf_dir=opts.variable_psf_dir
            )
    else:
        # Run on all available CPUs.
        from multiprocessing import Pool, cpu_count
        import itertools
        n_proc = cpu_count()
        pool = Pool(processes=n_proc)

        subfield_range = numpy.arange(200)
        iterator = itertools.izip(subfield_range,
                                  itertools.repeat(sim_dir),
                                  itertools.repeat(output_dir))                              
        R = pool.map(EstimateAllShearsStar,iterator)

 
if __name__ == "__main__":
    import pdb, traceback
    try:
        main(sys.argv)
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
