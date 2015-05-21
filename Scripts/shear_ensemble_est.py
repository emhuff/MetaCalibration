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
        path = path+'Outputs-Moments/cgc_metacal_moments*.fits'
    elif mc_type=='noaber-regauss-sym':
        path = path+'Outputs-Regauss-NoAber-SymNoise/cgc_noaber_metacal_symm*.fits'
    elif mc_type=='noaber-regauss':
        path = path+'Outputs-Regauss-NoAber/cgc_noaber_metacal*.fits'
    else:
        raise RuntimeError('Unrecognized mc_type: %s'%mc_type)

    catFiles = glob.glob(path)
    if len(catFiles) == 0:
        raise RuntimeError("No catalogs found with path %s!"%path)
    catalogs = []
    for thisFile in catFiles:
        catalogs.append( fits.getdata(thisFile) )

    return catalogs


def buildPrior(catalogs = None, nbins = 100):
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

    print '  About to build prior...'
    bin_edges, e1_prior_hist, e2_prior_hist, de1_dg, de2_dg = buildPrior(catalogs)
    print '  Done building prior, now doing rest of inference.'
    gamma1_raw = np.zeros(len(catalogs))
    gamma2_raw = np.zeros(len(catalogs))
    gamma1_opt = np.zeros(len(catalogs))
    gamma2_opt = np.zeros(len(catalogs))
    gamma1_var = np.zeros(len(catalogs))
    gamma2_var = np.zeros(len(catalogs))
    
    field_id = np.zeros(len(catalogs))
    psf_e1 = np.zeros(len(catalogs))
    psf_e2 = np.zeros(len(catalogs))
    
    covar1_scaled = - np.outer( e1_prior_hist, e1_prior_hist) * ( np.ones( (e1_prior_hist.size, e1_prior_hist.size) ) - np.diag(np.ones(e1_prior_hist.size) ) ) + np.diag( e1_prior_hist * (1 - e1_prior_hist) )
    covar2_scaled = - np.outer( e2_prior_hist, e2_prior_hist) * ( np.ones( (e2_prior_hist.size, e2_prior_hist.size) ) - np.diag(np.ones(e2_prior_hist.size) ) ) + np.diag( e2_prior_hist * (1 - e2_prior_hist) )    
    for catalog,i in zip(catalogs, xrange(len(catalogs) )):
        
        this_e1_hist, _ = np.histogram(catalog.g1 - catalog.c1 - catalog.a1*catalog.psf_e1 , bins = bin_edges )
        this_e1_hist = this_e1_hist * 1./catalog.size
        this_e2_hist, _ = np.histogram(catalog.g2 - catalog.c2 - catalog.a2*catalog.psf_e2, bins = bin_edges )
        this_e2_hist = this_e2_hist * 1./catalog.size
        # covar_hist = N_obj  * covar; but we divide hist by N_obj, so divide covar_hist by N_obj*N_obj
        this_covar1 = covar1_scaled * 1./catalog.size
        this_covar2 = covar2_scaled * 1./catalog.size
        this_cinv1 = np.linalg.pinv(this_covar1)
        this_cinv2 = np.linalg.pinv(this_covar2)

        gamma1_raw[i] = linear_estimator(data=this_e1_hist, null=e1_prior_hist, deriv=de1_dg)
        gamma2_raw[i] = linear_estimator(data=this_e2_hist, null=e2_prior_hist, deriv=de2_dg) 
        this_g1_opt, this_g1_var = \
            linear_estimator(data=this_e1_hist, null=e1_prior_hist, deriv=de1_dg, cinv=this_cinv1)
        this_g2_opt, this_g2_var = \
            linear_estimator(data=this_e2_hist, null=e2_prior_hist, deriv=de2_dg, cinv=this_cinv2) 
        gamma1_opt[i] = this_g1_opt
        gamma2_opt[i] = this_g2_opt
        gamma1_var[i] = this_g1_var
        gamma2_var[i] = this_g2_var

        field_id[i] = catalog[0]['id'] / 100000
        psf_e1[i] = catalog[0]['psf_e1']
        psf_e2[i] = catalog[0]['psf_e2']


    return field_id, gamma1_raw, gamma2_raw, gamma1_opt, gamma2_opt, gamma1_var, gamma2_var, psf_e1, psf_e2


def makePlots(field_id=None, g1=None, g2=None, err1 = None, err2 = None,
              psf_e1 = None, psf_e2 = None, truthFile = 'cgc-truthtable.txt', figName= None ):
    truthTable = np.loadtxt(truthFile, dtype = [('field_id',np.int), ('g1',np.double), ('g2',np.double ) ])
    
    obsTable = np.empty(field_id.size, [('field_id',np.int), ('g1',np.double), ('g2',np.double ),
                                        ('err1',np.double),('err2',np.double),
                                        ('psf_e1',np.double),('psf_e2',np.double)] )
    obsTable['field_id'] = field_id
    obsTable['g1'] = g1
    obsTable['g2'] = g2
    obsTable['psf_e1'] = psf_e1
    obsTable['psf_e2'] = psf_e2
    
    if (err1 is not None) and (err2 is not None):
        obsTable['err1'] = err1
        obsTable['err2'] = err2
        use_errors = True
    else:
        use_errors = False
    
    truthTable.sort(order='field_id')
    obsTable.sort(order='field_id')

    import matplotlib.pyplot as plt
    if not use_errors:
        fig,((ax1,ax2), (ax3,ax4)) = plt.subplots( nrows=2,ncols=2,figsize=(14,21) )
        ax1.plot(truthTable['g1'],obsTable['g1'],'.')
        ax1.plot(truthTable['g1'],truthTable['g1'],'--',color='red')
        ax1.set_title('g1')
        ax2.plot(truthTable['g2'],obsTable['g2'],'.')
        ax2.plot(truthTable['g2'],truthTable['g2'],'--',color='red')
        ax2.set_title('g2')

    
        ax3.plot(truthTable['g1'], obsTable['g1'] - truthTable['g1'],'.')
        ax3.axhline(0.,linestyle='--',color='red')
        ax3.set_ylim([-0.02,0.02])
        ax4.plot(truthTable['g2'], obsTable['g2'] - truthTable['g2'],'.')
        ax4.axhline(0.,linestyle='--',color='red')
        ax4.set_ylim([-0.02,0.02])
        fig.savefig(figName)
    else:
        fig,((ax1,ax2), (ax3,ax4), (ax5, ax6)) = plt.subplots( nrows=3,ncols=2,figsize=(14,21) )
        ax1.errorbar(truthTable['g1'],obsTable['g1'],obsTable['err1'],linestyle='.')
        ax1.plot(truthTable['g1'],truthTable['g1'],linestyle='--',color='red')
        ax1.set_title('g1')
        ax2.errorbar(truthTable['g2'],obsTable['g2'],obsTable['err2'],linestyle='.')
        ax2.plot(truthTable['g2'],truthTable['g2'],'--',color='red')
        ax2.set_title('g2')

    
        ax3.plot(truthTable['g1'], obsTable['g1'] - truthTable['g1'],'.',color='blue')
        ax3.axhline(0.,linestyle='--',color='red')
        ax3.axhspan(obsTable[0]['err1'],-obsTable[0]['err1'],alpha=0.2,color='red')
        ax3.set_ylim([-0.02,0.02])
        ax4.plot(truthTable['g2'], obsTable['g2'] - truthTable['g2'],'.',color='blue')
        ax4.axhline(0.,linestyle='--',color='red')
        ax4.axhspan(obsTable[0]['err1'],-obsTable[0]['err1'],alpha=0.2,color='red')        
        ax4.set_ylim([-0.02,0.02])

        ax5.plot(obsTable['psf_e1'], obsTable['g1'] - truthTable['g1'],'.',color='blue')
        ax5.axhline(0.,linestyle='--',color='red')
        ax5.axhspan(obsTable[0]['err1'],-obsTable[0]['err1'],alpha=0.2,color='red')
        ax5.set_xlim([-0.1,0.1])
        ax5.set_ylim([-0.02,0.02])
        ax5.set_title('psf trend (e1)')
        ax6.plot(obsTable['psf_e2'], obsTable['g2'] - truthTable['g2'],'.',color='blue')
        ax6.axhline(0.,linestyle='--',color='red')
        ax6.axhspan(obsTable[0]['err1'],-obsTable[0]['err1'],alpha=0.2,color='red')
        ax6.set_title('psf trend (e2)')
        ax6.set_xlim([-0.05,0.05])
        ax6.set_ylim([-0.02,0.02])
        fig.savefig(figName)

        

def main(args):

    # Set defaults and parse args.  This is kind of a stupid way to do it, since right now you can
    # specify either path, or path AND mc_type, or path AND mc_type AND outfile, but you can't (for
    # example) just specify outfile or just specify mc_type.  But it'll do for now.
    path = '../Great3/'
    mc_type = 'regauss'
    #truthFile = 'cgc-noaber-truthtable.txt'
    truthFile = 'cgc-truthtable.txt'
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
    field_id, g1raw, g2raw, g1opt, g2opt, g1var, g2var, psf_e1, psf_e2 = doInference(catalogs= catalogs)
    print 'Writing field_id, g1raw, g2raw, g1opt, g2opt, g1var,g2var to file %s'%outfile
    out_data = np.column_stack((field_id, g1raw, g2raw, g1opt, g2opt, g1var, g2var))
    np.savetxt(outfile, out_data, fmt='%i, %10.4e %10.4e %10.4e %10.4e %10.4e %10.4e')
    makePlots(field_id=field_id, g1=g1opt, g2=g2opt, err1 = np.sqrt(g1var), err2 = np.sqrt(g2var),
              psf_e1 = psf_e1, psf_e2 = psf_e2,
              truthFile = truthFile,figName=mc_type+'-opt-shear_plots')


if __name__ == "__main__":
    import pdb, traceback
    try:
        main(sys.argv)
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
