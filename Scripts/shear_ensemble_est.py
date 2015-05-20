#!/usr/bin/env python
import sys
import time
import os
import optparse
import numpy as np
import glob
from astropy.io import fits

def getAllCatalogs( path = '../Great3/', mc_type = None ):
    
    if mc_type=='regauss':
        path = path+'Outputs-Regauss/cgc_metacal_regauss_fix*.fits'
    elif mc_type=='regauss-sym':
        path = path+'Outputs-Regauss-SymNoise/cgc_metacal_symm*.fits'
    elif mc_type=='ksb':
        path = path+'Outputs-KSB/output_catalog*.fits'
    elif mc_type=='none-regauss':
        path = path+'Outputs-CGN-Regauss/cgc_metacal_moments*.fits'
    elif mc_type=='moments':
        path = path+'cgc_metacal_moments*.fits'
    elif mc_type=='noaber-regauss-sym':
        path = path+'Outputs-Regauss/cgc_noaber_metacal_symm*.fits'
    else:
        raise RuntimeError('Unrecognized mc_type: %s'%mc_type)

    catFiles = glob.glob(path)
    if len(catFiles) == 0:
        raise RuntimeError("No catalogs found with path %s!"%path)
    catalogs = []
    for thisFile in catFiles:
        catalogs.append( fits.getdata(thisFile) )

    return catalogs


def buildPrior(catalogs = None, nbins = 25):
    # Get a big master list of all the ellipticities in all fields.
    # Sadly you cannot retain column identity when using hstack, so we have to do the manipulations
    # for each catalog to get a list of e1 arrays to stack.
    e1_corr = []
    e2_corr = []
    r1 = []
    r2 = []
    for catalog in catalogs:
        e1_corr.append(catalog.g1 - catalog.c1 - catalog.a1*catalog.psf_e1)
        e2_corr.append(catalog.g2 - catalog.c2 - catalog.a2*catalog.psf_e2)
        r1.append(catalog.R1)
        r2.append(catalog.R2)
    e1_corr = np.hstack(e1_corr)
    e2_corr = np.hstack(e2_corr)
    r1 = np.hstack(r1)
    r2 = np.hstack(r2)
    e1prior = np.hstack( (e1_corr, -e1_corr ) )
    e2prior = np.hstack( (e2_corr, -e2_corr ) )
    all_e = np.hstack( (e1prior, e2prior))

    # Define bins.  np.percentile cannot take a list of percentile levels, so we have to stupidly
    # loop over the percentile levels we want.
    percentile_levels = np.linspace(0, 100, nbins)
    bin_edges = []
    for percentile_level in percentile_levels:
        bin_edges.append(np.percentile(all_e, percentile_level))
    bin_edges = np.array(bin_edges)
    bin_edges[0] = bin_edges[0] - 1.1*np.abs(bin_edges[0] )
    bin_edges[-1] = bin_edges[-1] + 1.1*np.abs(bin_edges[-1] )

    # Compute priors.
    e1_prior_hist, _ = np.histogram(e1prior, bins = bin_edges)
    e2_prior_hist, _ = np.histogram(e2prior, bins = bin_edges)

    e1_prior_hist = e1_prior_hist * 1./e1prior.size
    e2_prior_hist = e2_prior_hist * 1./e2prior.size
    
    # Compute derivatives.
    # Note from Rachel: changed code inside of the np.hstack() below.  I think it should be e1+r1*dg
    # and -e1+r1*dg, because regardless of whether e1 is >0 or <0, it should still be shifted to a
    # positive direction if dg>0.  Previous code had -(e1+r1*dg) which does the opposite, i.e.,
    # shifts e1 negative if dg is positive.
    dg = 0.01
    e1_prior_hist_mod, _  = np.histogram( 
        np.hstack( (e1_corr+r1*dg, -e1_corr+(r1*dg) ) ),  bins=bin_edges)
    e1_prior_hist_mod = e1_prior_hist_mod * 1./e1prior.size

    e2_prior_hist_mod, _  = np.histogram( 
        np.hstack( (e2_corr+r2*dg, -e2_corr+(r2*dg) ) ),  bins=bin_edges)
    e2_prior_hist_mod = e2_prior_hist_mod * 1./e2prior.size

    # Note from Rachel: updated the denominator to be 2*dg since we did a two-sided derivative.
    de1_dg = ( e1_prior_hist_mod - e1_prior_hist) / (2*dg)
    de2_dg = ( e2_prior_hist_mod - e2_prior_hist) / (2*dg)
    
    return bin_edges, e1_prior_hist, e2_prior_hist, de1_dg, de2_dg


def linear_estimator(data = None, null = None, deriv = None, cinv = None):
    if cinv is None:
        est= np.dot( (data - null), deriv) / np.dot( deriv, deriv )
        return est
    if cinv is not None:
        est = np.dot(np.dot( deriv.T, cinv), (data - null ) )/ (np.dot( np.dot( deriv.T, cinv) , deriv) )
        var = 1./ (np.dot( np.dot( deriv.T, cinv) , deriv) )
        return est, var

    
def doInference(catalogs= None):

    print '  About to build prior...'
    bin_edges, e1_prior_hist, e2_prior_hist, de1_dg, de2_dg = buildPrior(catalogs)
    print de1_dg
    print '  Done building prior, now doing rest of inference.'
    gamma1_raw = np.zeros(len(catalogs))
    gamma2_raw = np.zeros(len(catalogs))
    gamma1_opt = np.zeros(len(catalogs))
    gamma2_opt = np.zeros(len(catalogs))
    gamma1_var = np.zeros(len(catalogs))
    gamma2_var = np.zeros(len(catalogs))

    covar1_scaled = - np.outer( e1_prior_hist, e1_prior_hist) * ( np.ones( (e1_prior_hist.size, e1_prior_hist.size) ) - np.diag(np.ones(e1_prior_hist.size) ) ) + np.diag( e1_prior_hist * (1 - e1_prior_hist) )
    covar2_scaled = - np.outer( e2_prior_hist, e2_prior_hist) * ( np.ones( (e2_prior_hist.size, e2_prior_hist.size) ) - np.diag(np.ones(e2_prior_hist.size) ) ) + np.diag( e2_prior_hist * (1 - e2_prior_hist) )    
    
    
    for catalog,i in zip(catalogs, xrange(len(catalogs) )):
        
        this_e1_hist, _ = np.histogram(catalog.g1, bins = bin_edges )
        this_e1_hist = this_e1_hist / catalog.size
        this_e2_hist, _ = np.histogram(catalog.g2, bins = bin_edges )
        this_e2_hist = this_e2_hist / catalog.size
        # covar_hist = N_obj  * covar; but we divide hist by N_obj, so divide covar_hist by N_obj*N_obj
        this_covar1 = covar1_scaled / catalog.size 
        this_covar2 = covar2_scaled / catalog.size
        this_cinv1 = np.linalg.pinv(this_covar1)
        this_cinv2 = np.linalg.pinv(this_covar2)
        gamma1_raw[i] = linear_estimator( data = this_e1_hist, null = e1_prior_hist, deriv = de1_dg)
        gamma2_raw[i] = linear_estimator( data = this_e2_hist, null = e2_prior_hist, deriv = de2_dg) 
        this_g1_opt, this_g1_var = linear_estimator( data = this_e1_hist, null = e1_prior_hist, deriv = de1_dg, cinv = this_cinv1)
        this_g2_opt, this_g2_var = linear_estimator( data = this_e2_hist, null = e2_prior_hist, deriv = de2_dg, cinv = this_cinv2) 
        gamma1_opt[i] = this_g1_opt
        gamma2_opt[i] = this_g2_opt
        gamma1_var[i] = this_g1_var
        gamma2_var[i] = this_g2_var

    return gamma1_raw, gamma2_raw, gamma1_opt, gamma2_opt, gamma1_var, gamma2_var

def main(args):

    # Set defaults and parse args.  This is kind of a stupid way to do it, since right now you can
    # specify either path, or path AND mc_type, or path AND mc_type AND outfile, but you can't (for
    # example) just specify outfile or just specify mc_type.  But it'll do for now.
    path = '../Great3/'
    mc_type = 'regauss'
    outfile = 'tmp_outfile.txt'
    if len(args) > 1:
        if len(args) > 4:
            raise RuntimeError("I do not know how to handle that many arguments.")
        elif len(args) == 4:
            outfile = args[3]
        elif len(args) == 3:
            mc_type = args[2]
        path = args[1]

    print 'Getting catalogs from path %s and mc_type %s'%(path, mc_type)
    catalogs = getAllCatalogs(path=path, mc_type=mc_type)
    print 'Got %d catalogs, doing inference'%len(catalogs)
    g1raw, g2raw, g1opt, g2opt, g1var,g2var = doInference(catalogs= catalogs)
    print 'Writing g1raw, g2raw, g1opt, g2opt, g1var,g2var to file %s'%outfile
    out_data = np.column_stack((g1raw, g2raw, g1opt, g2opt, g1var, g2var))
    np.savetxt(outfile, out_data, fmt='%10.4e %10.4e %10.4e %10.4e %10.4e %10.4e')

if __name__ == "__main__":
    main(sys.argv)
