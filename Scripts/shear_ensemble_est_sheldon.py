#!/usr/bin/env python
import sys
import time
import os
import optparse
import numpy as np
import glob
from astropy.io import fits
import re
import galsim
import esutil

    
def getAllCatalogs( path = '/nfs/slac/des/fs1/g/sims/esheldon/lensing/great3reredux/', subsample = True, nrows = 100000 ):


    fields = ["mcal-v05s02/collated/mcal-v05s02.fits",\
              "mcal-v06s01/collated/mcal-v06s01.fits",\
              "mcal-v07s01/collated/mcal-v07s01.fits",\
              "mcal-v08s01/collated/mcal-v08s01.fits"]
    catalogs = []
    cat_dtype =  np.dtype([('id','>i8'),('g1','>f8'),('R1','>f8'),('a1','>f8'),('c1','>f8'), ('psf_e1','>f8'),('g2','>f8'),('R2','>f8'),('a2','>f8'),('c2','>f8'), ('psf_e2','>f8'),('weight','>f8')])
    for thisfield,field_id in zip(fields, np.arange(len(fields))):
        filename = path + thisfield
        if subsample is True:
            data = esutil.io.read(filename, rows=[np.arange(nrows)], \
                               columns=['exp_mcal_g','exp_mcal_R', 'exp_mcal_Rpsf','exp_mcal_gpsf','exp_mcal_c','exp_flags'], ext=1)
            keep = data['exp_flags'] == 0
            
        else:
            data = esutil.io.read(filename, \
                               columns=['exp_mcal_g','exp_mcal_R', 'exp_mcal_Rpsf','exp_mcal_gpsf','exp_mcal_c','exp_flags'], ext=1)
            keep = data['exp_flags'] == 0
        this_catalog = np.empty(np.sum(keep), dtype = cat_dtype)
        this_catalog['id'] = 1000000 * field_id 
        this_catalog['g1'] = data[keep]['exp_mcal_g'][:,0]
        this_catalog['g2'] = data[keep]['exp_mcal_g'][:,1]
        this_catalog['R1'] = data[keep]['exp_mcal_R'][:,0,0]
        this_catalog['R2'] = data[keep]['exp_mcal_R'][:,1,1]
        this_catalog['a1'] = data[keep]['exp_mcal_Rpsf'][:,0]
        this_catalog['a2'] = data[keep]['exp_mcal_Rpsf'][:,1]
        this_catalog['psf_e1'] = data[keep]['exp_mcal_gpsf'][:,0]
        this_catalog['psf_e2'] = data[keep]['exp_mcal_gpsf'][:,0]
        this_catalog['c1'] = data[keep]['exp_mcal_c'][:,0]
        this_catalog['c2'] = data[keep]['exp_mcal_c'][:,1]
        this_catalog['weight'] = np.zeros(np.sum(keep))+1.
        catalogs.append(this_catalog)
    return catalogs

    return catalogs, truthFile



def buildPrior(catalogs=None, nbins=100, bins = None, doplot = False, mc_type = None):
    # Get a big master list of all the ellipticities in all fields.
    # Sadly you cannot retain column identity when using hstack, so we have to do the manipulations
    # for each catalog to get a list of e1 arrays to stack.
    e1_corr = []
    e2_corr = []
    r1 = []
    r2 = []
    for catalog in catalogs:
        e1_corr.append(catalog['g1'] - catalog['c1'] - catalog['a1']*catalog['psf_e1'])
        e2_corr.append(catalog['g2'] - catalog['c2']- catalog['a2']*catalog['psf_e2'])
        r1.append(catalog['R1'])
        r2.append(catalog['R2'])
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
    e1_prior_hist_mod_p, _  = np.histogram(
        np.hstack( (e1_corr+r1*dg, -e1_corr+(r1*dg) ) ),  bins=bin_edges)
    e1_prior_hist_mod_p = e1_prior_hist_mod_p * 1./e1prior.size

    e1_prior_hist_mod_m, _  = np.histogram(
        np.hstack( (e1_corr-r1*dg, -e1_corr-(r1*dg) ) ),  bins=bin_edges)
    e1_prior_hist_mod_m = e1_prior_hist_mod_m * 1./e1prior.size
    #e1_prior_hist_mod = ( e1_prior_hist_mod_p - e1_prior_hist_mod_m ) /2.
    
    
    e2_prior_hist_mod_p, _  = np.histogram(
        np.hstack( (e2_corr+r2*dg, -e2_corr+(r2*dg) ) ),  bins=bin_edges)
    e2_prior_hist_mod_p = e2_prior_hist_mod_p * 1./e2prior.size

    e2_prior_hist_mod_m, _  = np.histogram(
        np.hstack( (e2_corr - r2*dg, -e2_corr - (r2*dg) ) ),  bins=bin_edges)
    e2_prior_hist_mod_m = e2_prior_hist_mod_m * 1./e2prior.size
    
    de1_dg = ( e1_prior_hist_mod_p - e1_prior_hist_mod_m) / (2*dg)
    de2_dg = ( e2_prior_hist_mod_p - e2_prior_hist_mod_m) / (2*dg)
    de1_dg[-1] = 0.
    de1_dg[0] = 0.
    de2_dg[-1] = 0.
    de2_dg[0] = 0.

    if doplot is True:
        import matplotlib.pyplot as plt
        fig,(ax1,ax2) =plt.subplots(nrows = 2, ncols = 1,figsize = (7,14))
        ax1.hist(  e1_corr[np.abs(e1_corr) <= 4], bins=100, normed=True, log = True )
        for x in bin_edges: ax1.axvline(x,color='red')
        ax1.set_xlim([-10,10])
        ax1.set_xlabel('e')
        ax1.set_ylabel('N(e)')
        ax1.set_xscale('symlog')
        ax2.plot(  de1_dg )
        ax2.set_xlabel(' bin number')
        ax2.set_ylabel('de / dg')
        ax2.axhline(0,color='red',linestyle='--')
        fig.savefig(mc_type+'-prior_derivs')

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

    
def doInference(catalogs=None, nbins=None, mean = False, plotFile = None):

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

    if plotFile is not None:
        from matplotlib.backends.backend_pdf import PdfPages
        import matplotlib.pyplot as plt
        pp = PdfPages(plotFile+".pdf")

    if plotFile is not None:
        linear_bin_edges = np.linspace(-1.5,1.5,100)
        linear_bin_centers = (linear_bin_edges[0:-1] + linear_bin_edges[1:])/2.
        _, prior_alt_e1_hist, prior_alt_e2_hist, _, _ = buildPrior(catalogs, bins=linear_bin_edges)
            
    for catalog,i in zip(catalogs, xrange(len(catalogs) )):


        if mean is False:
            this_e1_hist, _ = np.histogram(catalog['g1'] - catalog['c1'] - catalog['a1']*catalog['psf_e1'] , bins = bin_edges )
            this_e1_hist = this_e1_hist * 1./catalog.size
            this_e2_hist, _ = np.histogram(catalog['g2'] - catalog['c2'] - catalog['a2']*catalog['psf_e2'], bins = bin_edges )
            this_e2_hist = this_e2_hist * 1./catalog.size
        
            # covar_hist = N_obj  * covar; but we divide hist by N_obj, so divide covar_hist by N_obj*N_obj
            this_covar1 = covar1_scaled * 1./catalog.size
            this_covar2 = covar2_scaled * 1./catalog.size
    
            # Try making a covariance matrix from just this field?
            this_field_covar1 = ( - np.outer( this_e1_hist, this_e1_hist) * ( np.ones( (this_e1_hist.size, this_e1_hist.size) ) - np.diag(np.ones(this_e1_hist.size) ) ) + np.diag( this_e1_hist * (1 - this_e1_hist) ) ) / catalog.size
            this_field_covar2 =  (- np.outer( this_e2_hist, this_e2_hist) * ( np.ones( (this_e2_hist.size, this_e2_hist.size) ) - np.diag(np.ones(this_e2_hist.size) ) ) + np.diag( this_e2_hist * (1 - this_e2_hist) ) ) / catalog.size
            try:
                this_cinv1 = np.linalg.pinv(this_field_covar1)
                this_cinv2 = np.linalg.pinv(this_field_covar2)
            except:
                this_cinv1 = np.linalg.inv(this_field_covar1)
                this_cinv2 = np.linalg.inv(this_field_covar2)

            # Get derivatives for this shear field.
            #_, _, _, this_de1_dg, this_de2_dg = buildPrior([catalog], nbins=nbins, bins = bin_edges)

            gamma1_raw[i] = linear_estimator(data=this_e1_hist, null=e1_prior_hist, deriv=de1_dg)
            gamma2_raw[i] = linear_estimator(data=this_e2_hist, null=e2_prior_hist, deriv=de2_dg) 
            this_g1_opt, this_g1_var = \
                linear_estimator(data=this_e1_hist, null=e1_prior_hist, deriv= de1_dg, cinv=this_cinv1)
            this_g2_opt, this_g2_var = \
                linear_estimator(data=this_e2_hist, null=e2_prior_hist, deriv= de2_dg, cinv=this_cinv2)
            if plotFile is not None:
                fig, (ax1, ax2) = plt.subplots(nrows=1, ncols = 2, figsize = (14,7))
                _, this_e1_hist, this_e2_hist, _, _ = buildPrior(catalog, bins=linear_bin_edges)
                
                # Finally, make a version with the shear and psf effects subtracted off.
                
                
                ax1.plot(linear_bin_centers, prior_alt_e1_hist, label = 'e1 prior')
                ax1.plot(linear_bin_centers, this_e1_hist, label = 'this_e1')
                ax1.legend(loc='best')
                ax1.plot((bin_edges[0:-1] + bin_edges[1:])/2., de1_dg)
                ax1.plot((bin_edges[0:-1] + bin_edges[1:])/2., de2_dg)
                fig.savefig(pp, format="pdf")
                pp.close()
        elif mean is True:
            this_g1_opt =  np.average(catalog['g1'] - catalog['c1'] - catalog['a1'] * catalog['psf_e1'], weights = catalog['weight']) \
               / np.average(catalog['R1'], weights = catalog['weight'])
            this_g2_opt =  np.average(catalog['g2'] - catalog['c2'] - catalog['a2'] * catalog['psf_e2'], weights = catalog['weight']) \
              / np.average(catalog['R2'], weights = catalog['weight'])
            this_g1_var =  np.average(  ( ( catalog['g1'] - catalog['c1'] - catalog['a1'] * catalog['psf_e1']) - this_g1_opt )**2, weights = catalog['weight']) *1./ len(catalog)
            this_g2_var =  np.average(  ( ( catalog['g2'] - catalog['c2'] - catalog['a2'] * catalog['psf_e2']) - this_g2_opt )**2, weights = catalog['weight']) *1./ len(catalog)

                            
        gamma1_opt[i] = this_g1_opt
        gamma2_opt[i] = this_g2_opt
        gamma1_var[i] = this_g1_var
        gamma2_var[i] = this_g2_var

        e1_hist_desheared, _ = np.histogram(catalog['g1'] - catalog['R1'] * this_g1_opt - catalog['c1'] - catalog['a1']*catalog['psf_e1'] , bins = bin_edges )
        e1_hist_desheared = e1_hist_desheared * 1./catalog.size
        e2_hist_desheared, _ = np.histogram(catalog['g2'] - catalog['R2'] * this_g2_opt - catalog['c2'] - catalog['a2']*catalog['psf_e2'], bins = bin_edges )
        e2_hist_desheared = e2_hist_desheared * 1./catalog.size
        

        # Calculate the log-likelihood that this field was drawn from the shape distribution.
        field_e1_logL[i] = multinomial_logL(obs_hist= e1_hist_desheared * catalog.size, truth_prob = e1_prior_hist)
        field_e2_logL[i] = multinomial_logL(obs_hist= e2_hist_desheared * catalog.size, truth_prob = e2_prior_hist)

        

        field_id[i] = catalog[0]['id'] / 1000000
        psf_e1[i] = np.median(catalog['psf_e1'])
        psf_e2[i] = np.median(catalog['psf_e2'])


    return field_id, gamma1_raw, gamma2_raw, gamma1_opt, gamma2_opt, gamma1_var, gamma2_var, psf_e1, psf_e2, field_e1_logL, field_e2_logL



def makeFieldStructure(field_id=None, g1raw = None, g2raw = None, g1opt = None, g2opt = None, g1var = None, g2var = None,
                       psf_e1 = None, psf_e2 = None, e1_logL = None, e2_logL = None):
    field_str = np.empty(field_id.size, dtype=[('id',int),('g1raw',float), ('g2raw',float), ('g1opt',float), ('g2opt',float), ('g1var',float),('g2var',float), ('psf_e1',float), ('psf_e2',float), ('e1_logL',float), ('e2_logL',float)])
    field_str['id'] = field_id
    field_str['g1raw'] = g1raw
    field_str['g2raw'] = g2raw
    field_str['g1opt'] = g1opt
    field_str['g2opt'] = g2opt
    field_str['g1var'] = g1var
    field_str['g2var'] = g2var
    field_str['psf_e1'] = psf_e1
    field_str['psf_e2'] = psf_e2
    field_str['e1_logL'] = e1_logL
    field_str['e2_logL'] = e2_logL
    return field_str

    
def main(argv):

    import argparse

    description = """Analyze MetaCalibration outputs from Great3 and Great3++ simulations."""
    mc_choices =['g3redux']
    # Note: The above line needs to be consistent with the choices in getAllCatalogs.

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--path", dest="path", type=str, default="../Great3/",
                        help="path to MetaCalibration output catalogs")
    parser.add_argument("-mc","--mc_type", dest="mc_type", type=str, default="g3redux",
                        choices = mc_choices, help="metcalibration catalog type to use")
    parser.add_argument("-n", "--nbins", dest = "nbins", type = int, default= 20,
                        help = "number of bins to use in histogram estimator.")
    parser.add_argument("-o", "--outfile", dest = "outfile", type = str, default = "tmp_outfile.txt",
                        help = "destination for output per-field shear catalogs.")
    parser.add_argument("-p", "--percentile_cut", dest="percentile_cut",
                        help="percentile",type= float, default = 10)
    parser.add_argument("-dp", "--doplot", dest = "doplot", action="store_true")
    parser.add_argument("-a", "--do_all", dest = "do_all", action="store_true", default = False)
    parser.add_argument("-sn", "--snos_cut", dest="sn_cut",
                        help="percentile",type= float, default = 0)

    args = parser.parse_args(argv[1:])
    if args.sn_cut > 0:
        sn_cut = args.sn_cut
    else:
        sn_cut = None
        path = args.path

    if args.do_all is False:
        mc_type = args.mc_type
        nbins = args.nbins
        outfile = args.outfile
        print 'Getting catalogs from path %s and mc_type %s'%(path, mc_type)
        print 'Using %i bins for inference'% (nbins)
        catalogs = getAllCatalogs(subsample = True)
        print 'Got %d catalogs, doing inference'%len(catalogs)
        field_id, g1raw, g2raw, g1opt, g2opt, g1var, g2var, psf_e1, psf_e2, e1_logL, e2_logL = \
            doInference(catalogs=catalogs, nbins=nbins, mean=False, plotFile = "es_g3redux_histograms")
        field_str = makeFieldStructure(field_id=field_id, g1raw = g1raw, g2raw = g2raw, g1opt = g1opt, g2opt = g2opt,
                                    g1var = g1var, g2var = g2var, psf_e1 = psf_e1, psf_e2 = psf_e2,
                                    e1_logL = e1_logL, e2_logL = e2_logL)
        print 'Writing field_id, g1raw, g2raw, g1opt, g2opt, g1var, g2var, psf_e1, psf_qe2, e1_logL, e2_logL to file %s'%outfile
        out_data = np.column_stack((field_id, g1raw, g2raw, g1opt, g2opt, g1var, g2var, psf_e1, psf_e2, e1_logL, e2_logL))
        np.savetxt(outfile, out_data, fmt='%d %10.4e %10.4e %10.4e %10.4e %10.4e %10.4e %10.4e %10.4e %10.4e %10.4e')

        
if __name__ == "__main__":
    import pdb, traceback
    try:
        main(sys.argv)
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
