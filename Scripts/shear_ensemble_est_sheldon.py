#!/usr/bin/env python
import sys
import time
import os
import optparse
import numpy as np
import glob
from astropy.io import fits
import re
import esutil
import matplotlib.pyplot as plt

    
def getAllCatalogs( path = '/nfs/slac/des/fs1/g/sims/esheldon/lensing/great3reredux/', subsample = True, nrows = 100000 ):

    data = esutil.io.read("/nfs/slac/des/fs1/g/sims/esheldon/lensing/great3reredux/mcal-v10s02/collated/mcal-v10s02.fits")
    
    fields = np.unique(data['shear_index'])
    catalogs = []
    cat_dtype =  np.dtype([('id','>i8'),('g1','>f8'),('R1','>f8'),('a1','>f8'),('c1','>f8'),
                           ('psf_e1','>f8'),('g2','>f8'),('R2','>f8'),('a2','>f8'),('c2','>f8'),
                           ('psf_e2','>f8'),('weight','>f8')])
    for field_id in fields:
        keep = (data['flags'] == 0) & (data['shear_index'] == field_id) #& (data['pars'][:,5] > 20)
        this_catalog = np.empty(np.sum(keep), dtype = cat_dtype)
        this_catalog['id'] = 1000000 * field_id 
        this_catalog['g1'] = data[keep]['e'][:,0]
        this_catalog['g2'] = data[keep]['e'][:,1]
        this_catalog['R1'] = data[keep]['R'][:,0,0]
        this_catalog['R2'] = data[keep]['R'][:,1,1]
        this_catalog['a1'] = data[keep]['Rpsf'][:,0]
        this_catalog['a2'] = data[keep]['Rpsf'][:,1]
        this_catalog['psf_e1'] = data[keep]['epsf'][:,0]
        this_catalog['psf_e2'] = data[keep]['epsf'][:,1]
        this_catalog['c1'] = data[keep]['c'][:,0]
        this_catalog['c2'] = data[keep]['c'][:,1]
        this_catalog['weight'] = np.zeros(np.sum(keep))+1.
        catalogs.append(this_catalog)
    return catalogs

    return catalogs, truthFile



def reconstructMetacalMeas(g=None, R=None, a = None, c=None, psf_e=None, delta_g = 0.01 ):
    esum = 2*(c + g)
    ediff = 2*delta_g * R
    ep = (esum + ediff)/2. - a * psf_e
    em = (esum - ediff)/2. - a * psf_e
    return ep,em

def getHistogramDerivative(catalogs=None, bin_edges=None, delta_g = 0.01):
    e1_p_list = []
    e1_m_list = []
    e1_0_list = []
    
    e2_p_list = []
    e2_m_list = []
    e2_0_list = []
        
    for catalog in catalogs:
        e1p, e1m = reconstructMetacalMeas(g=catalog['g1'], R=catalog['R1'],
                                          a = catalog['a1'], c=catalog['c1'],
                                          psf_e=catalog['psf_e1'], delta_g = delta_g )
        e2p, e2m = reconstructMetacalMeas(g=catalog['g2'], R=catalog['R2'],
                                          a = catalog['a2'], c=catalog['c2'],
                                          psf_e=catalog['psf_e2'], delta_g = delta_g )
        e10 = catalog['g1'] - catalog['c1'] - catalog['a1']*catalog['psf_e1']
        e20 = catalog['g2'] - catalog['c2'] - catalog['a2']*catalog['psf_e2']
        e1_p_list.append(np.hstack((e1p)))
        e1_m_list.append(np.hstack((e1m)))
        e1_0_list.append(np.hstack((e10)))

        e2_p_list.append(np.hstack((e2p)))
        e2_m_list.append(np.hstack((e2m)))
        e2_0_list.append(np.hstack((e20)))
                
    e1_p = np.hstack(e1_p_list)
    e1_m = np.hstack(e1_m_list)
    e1_0 = np.hstack(e1_0_list)
    e2_p = np.hstack(e2_p_list)
    e2_m = np.hstack(e2_m_list)
    e2_0 = np.hstack(e2_0_list)
    
    h1_p, _ = np.histogram(e1_p, bins= bin_edges) 
    h1_m, _ = np.histogram(e1_m, bins= bin_edges) 
    h2_p, _ = np.histogram(e2_p, bins= bin_edges)
    h2_m, _ = np.histogram(e2_m, bins= bin_edges)

    dh1_dg1 = (h1_p - h1_m)/(2*delta_g)*1./len(e1_p)
    dh2_dg2 = (h2_p - h2_m)/(2*delta_g)*1./len(e2_p)

    return dh1_dg1, dh2_dg2
    

def buildPrior(catalogs=None, nbins=100, bins = None, doplot = False,
               mc_type = None, sym = False):
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
    if sym:
        e1prior = np.hstack( (e2_corr, -e2_corr ) )
        e2prior = np.hstack( (e2_corr, -e2_corr ) )
    else:
        e1prior = e1_corr
        e2prior = e2_corr
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

    de1_dg_nl, de2_dg_nl =  getHistogramDerivative(catalogs= catalogs, bin_edges=bin_edges, delta_g = dg)

    
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

    return bin_edges, e1_prior_hist, e2_prior_hist, de1_dg_nl, de2_dg_nl

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
        linear_bin_edges = np.linspace(-1.5,1.5,100)
        linear_bin_centers = (linear_bin_edges[0:-1] + linear_bin_edges[1:])/2.
        _, prior_alt_e1_hist, prior_alt_e2_hist, _, _ = buildPrior(catalogs, bins=linear_bin_edges)
            
    for catalog,i in zip(catalogs, xrange(len(catalogs) )):
        print "on catalog "+str(i)+" of  "+str(len(catalogs))

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
                fig, (ax1, ax2) = plt.subplots(nrows=1, ncols = 2, figsize = (21,7))
                _, this_e1_hist, this_e2_hist, _, _ = buildPrior(catalog, bins=linear_bin_edges,sym=False)
                e1_hist_desheared, _ = np.histogram(catalog['g1'] - catalog['R1'] * this_g1_opt - catalog['c1'] - catalog['a1']*catalog['psf_e1'] , bins = linear_bin_edges )
                e1_hist_desheared = e1_hist_desheared * 1./catalog.size

                # Finally, make a version with the shear and psf effects subtracted off.
                
                
                ax1.plot(linear_bin_centers, prior_alt_e1_hist, label = 'e1 prior')
                ax1.plot(linear_bin_centers, this_e1_hist, label = 'this_e1')
                ax1.plot(linear_bin_centers, e1_hist_desheared, label = 'e1_unsheared')

                ax1.legend(loc='best')
                ax2.plot(linear_bin_centers, this_e1_hist - prior_alt_e1_hist,label='this field - prior')
                ax2.plot(linear_bin_centers, e1_hist_desheared - prior_alt_e1_hist,label = 'desheared - prior')

                fig.savefig(pp, format="pdf")
        
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
    if plotFile is not None:
        pp.close()
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


def get_truthtable():
      shears= [ [-0.0369292, 0.0268976],
            [0.0147048, -0.023761],
            [0.010719, -0.0375317],
            [-0.0170452, 0.00550765],
            [-0.00558631, 0.0341976],
            [0.0100402, -0.0108189],
            [-0.0157668, 0.000980071],
            [-0.0053554, -0.0113379],
            [-0.00810751, 0.0124189],
            [0.0116076, 0.0380271],
            [-0.0221376, 0.00769596],
            [-0.031078, -0.00836606],
            [0.0128836, -0.0196881],
            [-0.0168117, 0.0199068],
            [-0.0092067, -0.0141776],
            [-0.022415, 0.00704573],
            [0.0362899, 0.00359092],
            [-0.00893076, -0.0480818],
            [-0.00770036, -0.0387106],
            [-0.0311143, 0.0270737],
            [0.0248045, -0.0192548],
            [-0.00803955, -0.00757901],
            [0.00551772, -0.0386444],
            [0.00700046, 0.0460475],
            [-0.0130024, 0.0151701],
            [-0.0424334, 0.0174551],
            [0.0330243, -0.00143821],
            [0.00406467, -0.0125254],
            [-0.00769266, -0.0286388],
            [-0.0159467, 0.00910803],
            [-0.0296337, -0.0106395],
            [-0.030306, -0.0148776],
            [-0.00192538, 0.0207825],
            [0.0157234, 0.0193959],
            [0.0101214, 0.0178025],
            [-0.000497735, -0.0127332],
            [-0.0264148, -0.016134],
            [-0.0276899, 0.00399226],
            [-0.0194067, 0.0217555],
            [-0.022896, 0.00584367],
            [-0.0295027, 0.0208863],
            [-0.0340295, -0.0202034],
            [-0.025543, -0.0393635],
            [0.013143, -0.0295915],
            [-0.00512104, -0.0114767],
            [0.0101185, 0.00367991],
            [-0.035196, -0.00340679],
            [0.0123071, -0.0247776],
            [0.0291862, 0.0130342],
            [-0.00992943, 0.0188574],
            [-0.0125323, 0.0414613],
            [0.0205224, 0.00919479],
            [-0.00197161, -0.0250597],
            [0.0308497, -0.00124479],
            [-0.0231097, 0.00355327],
            [-0.000815949, -0.0293916],
            [0.0365855, -0.0281216],
            [0.0298517, -0.0322181],
            [-0.00747514, 0.00995778],
            [0.0112657, -0.0155473],
            [0.0154795, -0.00174974],
            [0.00213608, -0.0451398],
            [-0.00887431, 0.0132027],
            [0.0200191, 0.0271031],
            [0.00613284, 0.0348119],
            [0.00918544, -0.0047391],
            [-0.026846, 0.0350538],
            [-0.0431593, 0.00481223],
            [-0.000893738, 0.0281559],
            [-0.0412704, 0.0246462],
            [0.00131108, -0.0164841],
            [-0.0122544, 0.00690147],
            [-0.0360282, -0.0169149],
            [0.0180157, 0.0305959],
            [-0.0314175, 0.0315025],
            [-0.0124494, 0.0308413],
            [0.0148659, -0.0476424],
            [0.00152103, -0.0232373],
            [0.0183979, 0.000250391],
            [0.0111579, -0.04835],
            [-0.0166408, 0.00402619],
            [-0.0165372, 0.0162025],
            [-0.033596, -0.0330116],
            [-0.027052, 0.0416133],
            [0.00920549, 0.0310317],
            [-0.00788643, 0.0214157],
            [0.0387487, -0.0169408],
            [0.0208807, 0.00832097],
            [0.0452373, 0.0113349],
            [0.00666435, 0.0124508],
            [-0.0423275, 0.00917404],
            [-0.0102854, -0.0317716],
            [-0.0364232, 0.0157652],
            [0.0238979, 0.0266593],
            [-0.0278488, -0.0214095],
            [0.0304696, -0.0125246],
            [-0.00272757, 0.0322831],
            [0.00018818, 0.0112566],
            [-0.010533, 0.00449835],
            [0.00243073, -0.0360685],
            [0.00388027, 0.023628],
            [-0.0163723, 0.0170477],
            [0.012608, 0.0230104],
            [0.0356393, -0.0086591],
            [-0.0112829, 0.00724424],
            [0.00816468, 0.0236215],
            [-0.00755304, -0.00835738],
            [0.00627764, 0.0111558],
            [0.0207231, 0.0245838],
            [0.0258988, 0.0398534],
            [-0.0178686, 0.00904728],
            [0.0350164, 0.00628305],
            [0.0248316, -0.0245096],
            [0.00684141, 0.0461624],
            [-0.00506305, -0.0154174],
            [0.0305498, 0.00160506],
            [-0.00489871, -0.0129169],
            [0.0265094, -0.0377505],
            [-0.0050039, 0.00921952],
            [-0.0354254, -0.000949451],
            [-0.00208154, 0.0477144],
            [0.00890316, 0.00884904],
            [0.0191448, -0.0227324],
            [0.0220497, 0.0441004],
            [-0.024574, 0.0347115],
            [0.00396406, -0.0136282],
            [-0.00760674, 0.0308806],
            [-0.0277704, 0.0386555],
            [-0.017493, -0.0175473],
            [0.0436661, -0.0027356],
            [0.0229195, -0.00907587],
            [0.0139287, -0.0389438],
            [0.00944163, -0.00476974],
            [-0.0270401, -0.024005],
            [0.0302651, 0.0297524],
            [0.00694205, 0.0360192],
            [-0.0106724, -0.0398671],
            [-0.0271398, 0.00056506],
            [0.00876183, -0.0123149],
            [-0.00598292, 0.0438725],
            [0.0288276, -0.0157463],
            [0.0380238, 0.0120442],
            [-0.0319324, -0.0296935],
            [-0.0030697, -0.0187077],
            [-0.0121803, -0.0173717],
            [0.0150902, 0.0446161],
            [0.0376233, -0.0220866],
            [-0.0147005, -0.0155701],
            [-0.017127, 0.0257343],
            [-0.0226136, -0.0263898],
            [-0.0217295, 0.0251977],
            [0.0215659, 0.0374364],
            [-0.0337836, -0.000711151],
            [0.00670888, 0.0362286],
            [0.0486262, -0.00743311],
            [-0.00202011, 0.0429544],
            [-0.0167753, -0.036627],
            [-0.0190894, -0.0306745],
            [-0.0136289, 0.00717367],
            [0.00448618, -0.048362],
            [0.0190139, -0.0322896],
            [0.0215585, 0.0439837],
            [0.0166828, 0.0288881],
            [0.00575044, -0.0158073],
            [0.0023268, 0.0124378],
            [-0.000502961, 0.0335579],
            [-0.020886, -0.00720564],
            [0.0192441, -0.0240871],
            [-0.00327226, -0.0181291],
            [-0.00051754, 0.0103705],
            [0.00248451, -0.016697],
            [0.0320086, -0.00997157],
            [0.0131062, -0.0111844],
            [0.000448852, -0.0115048],
            [0.0371046, 0.0272286],
            [-0.0373658, -0.0173048],
            [0.0333225, 0.00391339],
            [-0.0304504, -0.0151523],
            [0.0413634, 0.0136676],
            [-0.00857429, 0.0444579],
            [0.0255906, 0.0236618],
            [-0.0143247, 0.000978651],
            [-0.00394946, -0.0472592],
            [-0.0169541, 0.0106576],
            [0.00810509, 0.00746147],
            [-0.0333278, -0.00838693],
            [-0.0148629, 6.76447e-05],
            [0.00865976, -0.00870719],
            [-0.0119565, -0.00246735],
            [-0.027168, -0.011222],
            [0.0119151, -0.0267953],
            [0.00351119, -0.0106203],
            [0.014038, 0.00598558],
            [0.0248723, 0.023178],
            [-0.00424203, -0.0291179],
            [-0.0401158, 0.0040168],
            [-0.0101212, -0.0359837],
            [-0.0133273, -0.00303826],
            [0.00321895, 0.0226149],
            [0.0138849, -0.00272608],
            [0.00669208, -0.0181479],
            [0.00611157, -0.013983],
            [-0.0219695, 0.0356523],
            [0.0048154, -0.0125004],
            [-0.0287305, -0.0195992],
            [-0.0326577, 0.0347267],
            [0.00486702, -0.0259141],
            [0.032094, 0.016201],
            [0.0252234, -0.00177507],
            [0.0117135, 0.0256355],
            [0.0445831, -0.0194465],
            [0.00796167, -0.0426826],
            [-0.00342807, -0.0259635],
            [-0.0419965, -0.0236857],
            [0.0195201, -0.0328418],
            [-0.0150371, 0.0174543],
            [0.0227469, -0.03136],
            [0.0127359, -0.0124801],
            [0.0232993, 0.039482],
            [0.0213908, 0.0159259],
            [0.0110075, -0.0113531],
            [0.0376659, -0.0149542],
            [0.00100117, 0.0316909],
            [0.00586759, -0.0131346],
            [-0.00593623, -0.0185786],
            [-0.0230126, -0.0250201],
            [-0.014751, 0.00442692],
            [6.04729e-05, 0.0465425],
            [0.0222067, -0.0356898],
            [0.0179308, -0.00876186],
            [-0.0154091, -0.0214502],
            [-0.0142079, -0.0438975],
            [0.0141838, -0.00531064],
            [-0.0098439, -0.00633928],
            [0.00744103, 0.00947951],
            [0.0404729, -0.0168176],
            [-0.0112003, 0.0313309],
            [-0.0099825, 0.0296441],
            [0.0260437, 0.00230313],
            [-0.0464844, -0.00866735],
            [0.00839305, -0.0162292],
            [-0.000874439, 0.0179881],
            [-0.0132249, -0.000621603],
            [-0.00604973, -0.0395291],
            [0.0262383, -0.042439],
            [-0.00469083, 0.0104292],
            [0.0240346, 0.0388455],
            [0.011452, 0.0145279],
            [-0.0259977, 0.00467224],
            [0.00975905, 0.0240896],
            [-0.0423451, -0.0212828],
            [-0.0166085, -0.00220769],
            [0.0160108, -0.00732746],
            [0.0179268, 0.00231773],
            [-0.00729877, -0.0435186],
            [0.0244741, 0.0349244],
            [-0.0458469, -0.00973027],
            [-0.0279072, -0.0217365],
            [-0.0232985, -0.00797767],
            [-0.00161875, 0.0384378],
            [0.00215076, 0.0145467],
            [-0.0259101, 0.0153983],
            [-0.011385, 0.0137243],
            [-0.0136671, 0.00851378],
            [-0.023498, -0.00986002],
            [0.0373662, -0.00686131],
            [0.00394832, -0.0152173],
            [0.00205421, 0.040455],
            [-0.027321, -0.0150547],
            [-0.0253608, 0.0384098],
            [-0.00300706, 0.0229686],
            [0.0177499, 0.0116955],
            [0.0422454, -0.00869398],
            [0.0333173, 0.0351273],
            [0.00346382, 0.0151297],
            [0.0136908, 0.0191799],
            [0.0158374, 0.0111152],
            [-0.00488361, 0.02683],
            [0.0165917, 0.00371596],
            [-0.0183698, 0.0367385],
            [-0.0339046, 0.0218397],
            [-0.0479047, -0.0110466],
            [0.0135293, -0.0155816],
            [-0.0103649, 0.0103708],
            [-0.010204, 0.0183974],
            [0.0215688, -0.0234347],
            [0.0108064, 0.00136693],
            [-0.0487918, -0.00644605],
            [-0.039717, 0.0142356],
            [0.0372589, -0.0229965],
            [-0.0033006, 0.00987298],
            [-0.00751431, 0.0380412],
            [-0.00884511, 0.00791263],
            [-0.0398473, -0.0218551],
            [-0.0124897, 0.0082718],
            [0.0398795, -0.0125791],
            [-0.00779956, 0.0415062],
            [-0.0131707, 0.0245816],
            [0.00879533, -0.00504075],
            [-0.00544512, -0.00880617] ]
      shears_obj = np.empty(len(shears),dtype=[('g1',np.float),
                                               ('g2',np.float),
                                               ('field',np.int)])
      shears_arr = np.array(shears)
      shears_obj['g1'] = shears_arr[:,0]
      shears_obj['g2'] = shears_arr[:,1]
      shears_obj['field'] = np.arange(len(shears),dtype=np.int)
      return shears_obj

def doPlots(data,outfile = None):
    truthTable = get_truthtable()

    coeff1, covar1 = np.polyfit(truthTable['g1'],data['g1opt'] - truthTable['g1'],1,cov=True)
    coeff2, covar2 = np.polyfit(truthTable['g2'],data['g2opt'] - truthTable['g2'],1,cov=True)
    print 'm1 = '+str(coeff1[0])+'+/- '+str(np.sqrt(covar1[0,0]))+', c1 = '+str(coeff1[1])+'  '+str(np.sqrt(covar1[1,1]))
    print 'm2 = '+str(coeff2[0])+'+/- '+str(np.sqrt(covar2[0,0]))+', c2 = '+str(coeff2[1])+'  '+str(np.sqrt(covar2[1,1]))
    fig,((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(nrows=3,ncols=2,figsize=(14,7))
    ax1.plot(truthTable['g1'],data['g1opt'] - truthTable['g1'],'.')
    ax1.axhline(0,linestyle='--',color='red')
    ax1.plot(truthTable['g1'],coeff1[0]*truthTable['g1'] + coeff1[1],color='cyan')
    ax1.set_ylim(-0.02,0.02)
    ax2.plot(truthTable['g2'],data['g2opt'] - truthTable['g2'],'.')
    ax2.plot(truthTable['g2'],coeff2[0]*truthTable['g2'] + coeff2[1],color='cyan')
    ax2.axhline(0,linestyle='--',color='red')
    ax2.set_ylim(-0.02,0.02)

    ax3.plot(data['e1_logL'],data['g1opt'] - truthTable['g1'],'.')
    ax3.set_ylim(-0.02,0.02)
    ax3.axhline(0,linestyle='--',color='red')
    ax4.plot(data['e2_logL'],data['g2opt'] - truthTable['g2'],'.')
    ax4.set_ylim(-0.02,0.02)
    ax4.axhline(0,linestyle='--',color='red')
    
    ax5.plot(data['psf_e1'],data['g1opt'] - truthTable['g1'],'.')
    ax5.set_ylim(-0.02,0.02)
    ax5.axhline(0,linestyle='--',color='red')
    
    ax6.plot(data['psf_e2'],data['g2opt'] - truthTable['g2'],'.')
    ax6.axhline(0,linestyle='--',color='red')
    ax6.set_ylim(-0.02,0.02)

    fig.savefig(outfile)
    pass

  
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
                        help="percentile",type= float, default = 0)
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
            doInference(catalogs=catalogs, nbins=nbins, mean=False)
        field_str = makeFieldStructure(field_id=field_id, g1raw = g1raw, g2raw = g2raw, g1opt = g1opt, g2opt = g2opt,
                                    g1var = g1var, g2var = g2var, psf_e1 = psf_e1, psf_e2 = psf_e2,
                                    e1_logL = e1_logL, e2_logL = e2_logL)
        print 'Writing field_id, g1raw, g2raw, g1opt, g2opt, g1var, g2var, psf_e1, psf_qe2, e1_logL, e2_logL to file %s'%outfile
        out_data = np.column_stack((field_id, g1raw, g2raw, g1opt, g2opt, g1var, g2var, psf_e1, psf_e2, e1_logL, e2_logL))
        np.savetxt(outfile, out_data, fmt='%d %10.4e %10.4e %10.4e %10.4e %10.4e %10.4e %10.4e %10.4e %10.4e %10.4e')
        doPlots(field_str,outfile = 'default')
        
if __name__ == "__main__":
    import pdb, traceback
    try:
        main(sys.argv)
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
