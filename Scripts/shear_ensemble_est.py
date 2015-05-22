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
        truthFile = 'cgc-truthtable.txt'
    elif mc_type=='regauss-sym':
        path = path+'Outputs-Regauss-SymNoise/cgc_metacal_symm*.fits'
        truthFile = 'cgc-truthtable.txt'
    elif mc_type=='ksb':
        path = path+'Outputs-KSB/output_catalog*.fits'
        truthFile = 'cgc-truthtable.txt'
    elif mc_type=='none-regauss':
        path = path+'Outputs-CGN-Regauss/cgc_metacal_moments*.fits'
        truthFile = 'cgc-truthtable.txt'
    elif mc_type=='moments':
        path = path+'Outputs-Moments/cgc_metacal_moments*.fits'
        truthFile = 'cgc-truthtable.txt'
    elif mc_type=='noaber-regauss-sym':
        path = path+'Outputs-Regauss-NoAber-SymNoise/cgc_noaber_metacal_symm*.fits'
        truthFile = 'cgc-noaber-truthtable.txt'
    elif mc_type=='noaber-regauss':
        path = path+'Outputs-Regauss-NoAber/cgc_noaber_metacal*.fits'
        truthFile = 'cgc-noaber-truthtable.txt'
    elif mc_type == 'rgc-regauss':
        path = path+'Outputs-Real-Regauss/rgc_metacal-*.fits'
        truthFile = 'rgc-dummytable.txt'
    elif mc_type == 'rgc-noaber-regauss':
        path = path+'Outputs-Real-NoAber-Regauss/rgc_noaber_metacal*.fits'
        truthFile = 'rgc-noaber-dummytable.txt' 
    else:
        raise RuntimeError('Unrecognized mc_type: %s'%mc_type)

    

    catFiles = glob.glob(path)
    if len(catFiles) == 0:
        raise RuntimeError("No catalogs found with path %s!"%path)
    catalogs = []
    for thisFile in catFiles:
        catalogs.append( fits.getdata(thisFile) )

    return catalogs, truthFile



def buildPrior(catalogs=None, nbins=100, bins = None):
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
    if bins is None:
        percentile_levels = np.linspace(0, 100, nbins)
        bin_edges = []
        for percentile_level in percentile_levels:
            bin_edges.append(np.percentile(all_e, percentile_level))
        bin_edges = np.array(bin_edges)
        bin_edges[0] = bin_edges[0] - 1.1*np.abs(bin_edges[0] )
        bin_edges[-1] = bin_edges[-1] + 1.1*np.abs(bin_edges[-1] )
    else:
        bin_edges = bins

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

def multinomial_logL(obs_hist= None, truth_prob = None):
    # Make liberal use of Stirling's approx.
    Ntot = np.sum(obs_hist)
    log_Ntot_fac = Ntot * np.log(Ntot) - Ntot
    log_Ni_fac = np.sum( obs_hist * np.log(obs_hist) - obs_hist)
    log_prob = np.sum( obs_hist * np.log(truth_prob))
    like = log_Ntot_fac - log_Ni_fac + log_prob
    return like


def linear_estimator(data = None, null = None, deriv = None, cinv = None):
    if cinv is None:
        est= np.dot( (data - null), deriv) / np.dot( deriv, deriv )
        return est
    if cinv is not None:
        est = np.dot(np.dot( deriv.T, cinv), (data - null ) )/ (np.dot( np.dot( deriv.T, cinv) , deriv) )
        var = 1./ (np.dot( np.dot( deriv.T, cinv) , deriv) )
        return est, var

    
def doInference(catalogs=None, nbins=None):

    print '  About to build prior...'
    bin_edges, e1_prior_hist, e2_prior_hist, de1_dg, de2_dg = \
        buildPrior(catalogs, nbins=nbins)
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
    field_e1_logL = np.zeros(len(catalogs) )
    field_e2_logL = np.zeros(len(catalogs) )
    
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
    
        # Try making a covariance matrix from just this field?
        this_field_covar1 = ( - np.outer( this_e1_hist, this_e1_hist) * ( np.ones( (this_e1_hist.size, this_e1_hist.size) ) - np.diag(np.ones(this_e1_hist.size) ) ) + np.diag( this_e1_hist * (1 - this_e1_hist) ) ) / catalog.size
        this_field_covar2 =  (- np.outer( this_e2_hist, this_e2_hist) * ( np.ones( (this_e2_hist.size, this_e2_hist.size) ) - np.diag(np.ones(this_e2_hist.size) ) ) + np.diag( this_e2_hist * (1 - this_e2_hist) ) ) / catalog.size
        this_cinv1 = np.linalg.pinv(this_field_covar1)
        this_cinv2 = np.linalg.pinv(this_field_covar2)

        # Get derivatives for this shear field.
        #_, _, _, this_de1_dg, this_de2_dg = buildPrior([catalog], nbins=nbins, bins = bin_edges)

        gamma1_raw[i] = linear_estimator(data=this_e1_hist, null=e1_prior_hist, deriv=de1_dg)
        gamma2_raw[i] = linear_estimator(data=this_e2_hist, null=e2_prior_hist, deriv=de2_dg) 
        this_g1_opt, this_g1_var = \
            linear_estimator(data=this_e1_hist, null=e1_prior_hist, deriv= de1_dg, cinv=this_cinv1)
        this_g2_opt, this_g2_var = \
            linear_estimator(data=this_e2_hist, null=e2_prior_hist, deriv= de2_dg, cinv=this_cinv2) 
        gamma1_opt[i] = this_g1_opt
        gamma2_opt[i] = this_g2_opt
        gamma1_var[i] = this_g1_var
        gamma2_var[i] = this_g2_var

        e1_hist_desheared, _ = np.histogram(catalog.g1 - catalog.R1 * this_g1_opt - catalog.c1 - catalog.a1*catalog.psf_e1 , bins = bin_edges )
        e1_hist_desheared = e1_hist_desheared * 1./catalog.size
        e2_hist_desheared, _ = np.histogram(catalog.g2 - catalog.R2 * this_g2_opt - catalog.c2 - catalog.a2*catalog.psf_e2, bins = bin_edges )
        e2_hist_desheared = e2_hist_desheared * 1./catalog.size
        

        # Calculate the log-likelihood that this field was drawn from the shape distribution.
        field_e1_logL[i] = multinomial_logL(obs_hist= e1_hist_desheared * catalog.size, truth_prob = e1_prior_hist)
        field_e2_logL[i] = multinomial_logL(obs_hist= e2_hist_desheared * catalog.size, truth_prob = e2_prior_hist)

        

        field_id[i] = catalog[0]['id'] / 1000000
        psf_e1[i] = catalog[0]['psf_e1']
        psf_e2[i] = catalog[0]['psf_e2']


    return field_id, gamma1_raw, gamma2_raw, gamma1_opt, gamma2_opt, gamma1_var, gamma2_var, psf_e1, psf_e2, field_e1_logL, field_e2_logL


def makePlots(field_id=None, g1=None, g2=None, err1 = None, err2 = None,
              psf_e1 = None, psf_e2 = None, e1_logL = None, e2_logL = None,
              truthFile = 'cgc-truthtable.txt', figName= None ):
    truthTable = np.loadtxt(truthFile, dtype = [('field_id',np.int), ('g1',np.double), ('g2',np.double ) ])
    
    obsTable = np.empty(field_id.size, [('field_id',np.int), ('g1',np.double), ('g2',np.double ),
                                        ('err1',np.double),('err2',np.double),
                                        ('psf_e1',np.double),('psf_e2',np.double),
                                        ('e1_logL',np.double),('e2_logL',np.double)])
    obsTable['field_id'] = field_id
    obsTable['g1'] = g1
    obsTable['g2'] = g2
    obsTable['psf_e1'] = psf_e1
    obsTable['psf_e2'] = psf_e2
    obsTable['e1_logL'] = e1_logL
    obsTable['e2_logL'] = e2_logL
    
    if (err1 is not None) and (err2 is not None):
        obsTable['err1'] = err1
        obsTable['err2'] = err2
        use_errors = True
    else:
        use_errors = False


    truthTable.sort(order='field_id')
    obsTable.sort(order='field_id')
    shear_range = 2*( np.percentile( np.concatenate( (g1, g2) ), 75) - np.percentile( np.concatenate( (g1, g2) ), 50))

    
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
        fig,((ax1,ax2), (ax3,ax4), (ax5, ax6), (ax7,ax8)) = plt.subplots( nrows=4,ncols=2,figsize=(14,28) )
        ax1.errorbar(truthTable['g1'],obsTable['g1'],obsTable['err1'],linestyle='.')
        ax1.plot(truthTable['g1'],truthTable['g1'],linestyle='--',color='red')
        ax1.set_title('g1')
        ax2.errorbar(truthTable['g2'],obsTable['g2'],obsTable['err2'],linestyle='.')
        ax2.plot(truthTable['g2'],truthTable['g2'],'--',color='red')
        ax2.set_title('g2')

    
        ax3.plot(truthTable['g1'], obsTable['g1'] - truthTable['g1'],'.',color='blue')
        ax3.axhline(0.,linestyle='--',color='red')
        ax3.axhspan(obsTable[0]['err1'],-obsTable[0]['err1'],alpha=0.2,color='red')
        ax3.set_ylim([-shear_range, shear_range])
        ax4.plot(truthTable['g2'], obsTable['g2'] - truthTable['g2'],'.',color='blue')
        ax4.axhline(0.,linestyle='--',color='red')
        ax4.axhspan(obsTable[0]['err1'],-obsTable[0]['err1'],alpha=0.2,color='red')        
        ax4.set_ylim([-shear_range, shear_range])

        ax5.plot(obsTable['e1_logL'], obsTable['g1'] - truthTable['g1'],'.',color='blue')
        ax5.set_xlabel('multinomial log likelihood')
        ax5.set_ylabel('shear error (e1)')
        ax5.set_xscale('symlog')
        ax5.axhspan(np.median(obsTable['err1']),-np.median(obsTable['err1']),alpha=0.2,color='red')
        
        ax6.plot(obsTable['e2_logL'], obsTable['g2'] - truthTable['g2'],'.',color='blue')
        ax6.axhspan(np.median(obsTable['err1']),-np.median(obsTable['err1']),alpha=0.2,color='red')
        ax6.set_xlabel('multinomial log likelihood')
        ax6.set_ylabel('shear error (e2)')
        ax6.set_xscale('symlog')
        
        ax7.plot(obsTable['psf_e1'], obsTable['g1'] - truthTable['g1'],'.',color='blue')
        ax7.axhline(0.,linestyle='--',color='red')
        ax7.axhspan(np.median(obsTable['err1']),-np.median(obsTable['err1']),alpha=0.2,color='red')
        ax7.set_xlim([-0.04,0.04])
        ax7.set_ylim([-shear_range, shear_range])
        ax7.set_title('psf trend (e1)')
        ax8.plot(obsTable['psf_e2'], obsTable['g2'] - truthTable['g2'],'.',color='blue')
        ax8.axhline(0.,linestyle='--',color='red')
        ax8.axhspan(np.median(obsTable['err1']),-np.median(obsTable['err1']),alpha=0.2,color='red')
        ax8.set_title('psf trend (e2)')
        ax8.set_xlim([-0.04,0.04])
        ax8.set_ylim([-shear_range, shear_range])
        fig.savefig(figName)

        

def main(argv):

    # Set defaults and parse args.  This is kind of a stupid way to do it, since right now you can
    # specify either path, or path AND mc_type, or path AND mc_type AND outfile, but you can't (for
    # example) just specify outfile or just specify mc_type.  But it'll do for now.


    import argparse

    description = """Analyze MetaCalibration outputs from Great3 and Great3++ simulations."""
    mc_choices =['regauss', 'regauss-sym', 'ksb', 'none-regauss', 'moments', 'noaber-regauss-sym', 'noaber-regauss','rgc-regauss','rgc-noaber-regauss']
    # Note: The above line needs to be consistent with the choices in getAllCatalogs.
    
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--path", dest="path", type=str, default="../Great3/",
                        help="path to MetaCalibration output catalogs")
    parser.add_argument("-mc","--mc_type", dest="mc_type", type=str, default="regauss",
                        choices = mc_choices, help="metcalibration catalog type to use")
    parser.add_argument("-n", "--nbins", dest = "nbins", type = int, default= 80,
                        help = "number of bins to use in histogram estimator.")
    parser.add_argument("-o", "--outfile", dest = "outfile", type = str, default = "tmp_outfile.txt",
                        help = "destination for output per-field shear catalogs.")
    parser.add_argument("-dp", "--doplot", dest = "doplot", action="store_true")
    args = parser.parse_args(argv[1:])
    
    path = args.path
    mc_type = args.mc_type
    nbins = args.nbins
    outfile = args.outfile
    print 'Getting catalogs from path %s and mc_type %s'%(path, mc_type)
    print 'Using %i bins for inference'% (nbins)
    catalogs, truthfile = getAllCatalogs(path=path, mc_type=mc_type)
    print 'Got %d catalogs, doing inference'%len(catalogs)
    field_id, g1raw, g2raw, g1opt, g2opt, g1var, g2var, psf_e1, psf_e2, e1_logL, e2_logL = \
        doInference(catalogs=catalogs, nbins=nbins)
    print 'Writing field_id, g1raw, g2raw, g1opt, g2opt, g1var, g2var, psf_e1, psf_e2, e1_logL, e2_logL to file %s'%outfile
    out_data = np.column_stack((field_id, g1raw, g2raw, g1opt, g2opt, g1var, g2var, psf_e1, psf_e2, e1_logL, e2_logL))
    np.savetxt(outfile, out_data, fmt='%d %10.4e %10.4e %10.4e %10.4e %10.4e %10.4e %10.4e %10.4e %10.4e %10.4e')
    if args.doplot:
        print "Making plots..."
        makePlots(field_id=field_id, g1=g1opt, g2=g2opt, err1 = np.sqrt(g1var), err2 = np.sqrt(g2var),
                  psf_e1 = psf_e1, psf_e2 = psf_e2, e1_logL = e1_logL, e2_logL = e2_logL,
                  truthFile = truthfile,figName=mc_type+'-opt-shear_plots')
        print "wrote plots to "+mc_type+'-opt-shear_plots.png'


if __name__ == "__main__":
    import pdb, traceback
    try:
        main(sys.argv)
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
