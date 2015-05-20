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
    master = np.hstack(catalogs)
    e1_corr = master.g1 - master.c1 - master.a1 * master.psf_e1
    e2_corr = master.g2 - master.c2 - master.a2 * master.psf_e2
    e1prior = np.hstack( (e1_corr, -e1_corr ) )
    e2prior = np.hstack( (e2_corr, -e2_corr ) )
    all_e = np.hstack( (e1prior, e2prior))
    # Define bins.
    bin_edges = np.percentile( alle, np.linspace(0,100,nbins ) )
    bin_edges[0] = bin_edges[0] - 1.1*np.abs(bin_edges[0] )
    bin_edges[-1] = bin_edgs[-1] + 1.1*np.abs(bin_edges[-1] )

    # Compute priors.
    e1_prior_hist, _ = np.histogram(e1prior, bins = bin_edges)
    e2_prior_hist, _ = np.histogram(e2prior, bins = bin_edges )

    e1_prior_hist = e1_prior_hist * 1./e1prior.size
    e2_prior_hist = e2_prior_hist * 1./e2prior.size
    
    # Compute derivatives.
    dg1 = 0.01
    e1_prior_hist_mod, _  = np.histogram( np.hstack( ( e1_corr + master.r1 * dg, - (e1_corr + master.r1 * dg) ) ),  bins = bin_edges )
    e1_prior_hist_mod = e1_prior_hist_mod / (e1prior.size)

    e2_prior_hist_mod, _  = np.histogram( np.hstack( ( e2_corr + master.r1 * dg, - (e2_corr + master.r1 * dg) ) ),  bins = bin_edges )
    e2_prior_hist_mod = e2_prior_hist_mod / (e2prior.size)

    de1_dg = ( e1_prior_hist_mod - e1_prior_hist) / dg    
    de2_dg = ( e2_prior_hist_mod - e2_prior_hist) / dg
    
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

    bin_edges, e1_prior_hist, e2_prior_hist, de1_dg, de2_dg = buildPrior(catalogs)
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
        this_e1_hist = this_e1_hist * 1./catalog.size
        # covar_hist = N_obj  * covar; but we divide hist by N_obj, so divide covar_hist by N_obj*N_obj
        this_covar =  covar_scaled * 1./catalog.size 
        this_cinv = np.linalg.pinv(this_covar)
        gamma1_raw[i] = linear_estimator( data = this_e1_hist, null = e1_prior_hist, deriv = de1_dg)
        gamma2_raw[i] = linear_estimator( data = this_e2_hist, null = e2_prior_hist, deriv = de2_dg) 
        this_g1_opt, this_g1_var = linear_estimator( data = this_e1_hist, null = e1_prior_hist, deriv = de1_dg, cinv = this_cinv)
        this_g2_opt, this_g2_var = linear_estimator( data = this_e2_hist, null = e2_prior_hist, deriv = de2_dg, cinv = this_cinv) 
        gamma1_opt[i] = this_g1_opt
        gamma2_opt[i] = this_g2_opt
        gamma1_var[i] = this_g1_var
        gamma2_var[i] = this_g2_var

    return gamma1_raw, gamma2_raw, gamma1_opt, gamma2_opt, gamma1_var, gamma2_var

def main(args):

    # Set defaults and parse args.
    path = '../Great3/'
    mc_type = 'regauss'
    if len(args) > 1:
        if len(args) > 3:
            raise RuntimeError("I do not know how to handle that many arguments.")
        elif len(args) == 3:
            mc_type = args[2]
        path = args[1]

    catalogs = getAllCatalogs(path=path, mc_type=mc_type)
    g1raw, g2raw, g1opt, g2opt, g1var,g2var = doInference(catalogs= catalogs)

if __name__ == "__main__":
    main(sys.argv)
