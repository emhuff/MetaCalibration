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
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import em_student_t as emt


def getAllCatalogs( path = '/nfs/slac/des/fs1/g/sims/esheldon/lensing/great3reredux/', subsample = True, nrows = 100000 ):

    data = esutil.io.read("/nfs/slac/des/fs1/g/sims/esheldon/lensing/great3reredux/mcal-v10s02/collated/mcal-v10s02.fits")
    
    fields = np.unique(data['shear_index'])
    catalogs = []
    cat_dtype =  np.dtype([('id','>i8'),('g1','>f8'),('R1','>f8'),('a1','>f8'),('c1','>f8'),
                           ('psf_e1','>f8'),('g2','>f8'),('R2','>f8'),('a2','>f8'),('c2','>f8'),
                           ('psf_e2','>f8'),('weight','>f8')])
    for field_id in fields:
        keep = (data['flags'] == 0) & (data['shear_index'] == field_id) & (data['pars'][:,5] > 15)
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

def shear_avg(e, weights = True, n= 4, exp=False, scale = .25, pars = None):
    if weights is None:
        return np.mean(e)
    else:
        if n is not None:
            #weight = 1./(scale**n + np.abs(e)**n)
            weight = 1./(1 + (e/ scale)**2/n)**((n+1)/2.)
        elif exp is True:
            weight = np.exp(- 0.5*np.abs(e/scale)**2)
        elif pars is not None:
            weight = emt.t(e, 0., pars[1], pars[2])

        for i in xrange(10):
            eavg = np.average(e,weights=weight)
            if n is not None:
                weight = 1./(1 + ((e-eavg)/scale)**2/n)**((n+1)/2.)
            elif exp is True:
                weight = np.exp(- 0.5*(np.abs(e-eavg)/scale)**2)
            elif pars is not None:
                weight = emt.t(e - eavg, 0., pars[1], pars[2])

        logL = np.average(np.log(weight[weight > 0]) - np.log(np.sum(weight)))
        return np.average(e,weights=weight), logL

def shear_em(e):
    mu, sigma, nu = emt.estimate_t(e)
    return mu, sigma, nu



def shear_est(catalogs, truthFile, delta_g = 0.01, weights = True,mc_type=None):

    est1 = []
    est2 = []
    est1_err = []
    est2_err = []
    logL_e1 = []
    logL_e2 = []

    e1_master = np.hstack([catalog['g1'] - catalog['c1'] - catalog['a1']*catalog['psf_e1'] for catalog in catalogs])
    e2_master = np.hstack([catalog['g2'] - catalog['c2'] - catalog['a2']*catalog['psf_e2'] for catalog in catalogs])
    mu1, sigma1, nu1 = shear_em(e1_master)
    mu2, sigma2, nu2 = shear_em(e2_master)

    sigma1_global = sigma1
    sigma2_global = sigma2
    nu1_global = nu1
    nu2_global = nu2
    eps = .0
    
    bins = np.linspace(-10,10,250)
    xx = (bins[0:-1] + bins[1:])/2.
    yy_est =  emt.t(xx, mu1, sigma1, nu1)
    yy_est = yy_est/np.sum(yy_est)
    fig,(ax1,ax2) = plt.subplots(ncols=2,nrows=1,figsize=(14,7))
    yy_data,b = np.histogram(e1_master,bins=bins)
    yy_gauss = np.exp(-(xx-mu1)**2/2./sigma1**2)
    yy_gauss = yy_gauss/np.sum(yy_gauss)
    yy_power = 1./(1 + (xx/(sigma1/2.))**2/(nu1+eps))**(((nu1+eps)+1)/2.)
    yy_power = yy_power/np.sum(yy_power)
    ax1.plot(xx,yy_est,label='model')
    ax1.plot(xx,yy_data*1./np.sum(yy_data),label='data')
    ax1.plot(xx,yy_gauss*1./np.sum(yy_gauss),label='gauss')
    ax1.plot(xx,yy_power*1./np.sum(yy_power),label='weight')
    ax1.legend(loc='best')
    ax1.set_yscale('log')
    ax1.set_ylim(1e-6,1)
    ax1.annotate('mu = %.4f \n sigma = %.4f \n nu = %.4f'%(mu1, sigma1, nu1),xy=(0.05, 0.85), xycoords='axes fraction')

    yy_data,b = np.histogram(e2_master,bins=bins)
    yy_data = yy_data*1./np.sum(yy_data)
    yy_gauss = np.exp(-(xx-mu2)**2/2./sigma2**2)
    yy_gauss = yy_gauss/np.sum(yy_gauss)
    yy_power = 1./(1 + (xx/(sigma2/2.))**2/(nu2+eps))**(((nu2+eps)+1)/2.)
    yy_power = yy_power/np.sum(yy_power)
    ax2.plot(xx,yy_est,label='model')
    ax2.plot(xx,yy_data,label='data')
    ax2.plot(xx,yy_gauss,label='gauss')
    ax2.plot(xx,yy_power,label='weight')
    ax2.legend(loc='best')
    ax2.set_yscale('log')
    ax2.set_ylim(1e-6,1)
    ax2.annotate('mu = %.4f \n sigma = %.4f \n nu = %.4f'%(mu2, sigma2, nu2),xy=(0.05, 0.85), xycoords='axes fraction')

        
    fig.savefig(mc_type+'-prior_model_comparison')

    
    for i,catalog in enumerate(catalogs):
        if (i % 10) == 0:
            print "starting iter: "+str(i)+" of "+str(len(catalogs))
        e1p, e1m = reconstructMetacalMeas(g=catalog['g1'], R=catalog['R1'],
                                          a = catalog['a1'], c=catalog['c1'],
                                          psf_e=catalog['psf_e1'], delta_g = delta_g )
        e2p, e2m = reconstructMetacalMeas(g=catalog['g2'], R=catalog['R2'],
                                          a = catalog['a2'], c=catalog['c2'],
                                          psf_e=catalog['psf_e2'], delta_g = delta_g )
        e10 = catalog['g1'] - catalog['c1'] - catalog['a1']*catalog['psf_e1']
        e20 = catalog['g2'] - catalog['c2'] - catalog['a2']*catalog['psf_e2']

        mu1, sigma1, nu1 = shear_em(e10)
        mu2, sigma2, nu2 = shear_em(e20)

        _, logL1 = shear_avg(e10,n=nu1_global, scale = sigma1_global)
        _, logL2 = shear_avg(e20,n=nu2_global, scale = sigma2_global)
        
        g1p, _ = shear_avg(e1p,n=nu1+eps, scale = sigma1/2.)
        g10, _ = shear_avg(e10,n=nu1+eps, scale = sigma1/2.)
        g1m, _ = shear_avg(e1m,n=nu1+eps, scale = sigma1/2.)

        g2p, _ = shear_avg(e2p,n=nu2+eps, scale = sigma2/2.)
        g20, _ = shear_avg(e20,n=nu2+eps, scale = sigma2/2.)
        g2m, _ = shear_avg(e2m,n=nu2+eps, scale = sigma2/2.)

        m1 = (g1p - g1m)/(2*delta_g)
        c1 = (g1p + g1m)/2. - g10 
        m2 = (g2p - g2m)/(2*delta_g)
        c2 = (g2p + g2m)/2. - g20
        
        est1.append((g10 - c1)/m1)
        est2.append((g20 - c2)/m2)

        logL_e1.append(logL1)
        logL_e2.append(logL2)
        #logL_e1.append(-1.)
        #logL_e2.append(-1.)


            
        #this_err1 = np.std(catalog['g1'])/np.sqrt(catalog.size)
        #this_err2 = np.std(catalog['g2'])/np.sqrt(catalog.size)
        this_err1 = nu1*1./(nu1-2.)/np.sqrt(catalog.size)
        this_err2 = nu2*1./(nu2-2.)/np.sqrt(catalog.size)

        est1_err.append(this_err1/m1)
        est2_err.append(this_err2/m2)

    
    results = np.empty(len(catalogs), dtype = [('g1_est',np.float),('g2_est',np.float),
                                               ('g1_err',np.float),('g2_err',np.float),
                                               ('g1_raw',np.float),('g2_raw',np.float),
                                               ('g1_true',np.float),('g2_true',np.float),
                                               ('psf_e1',np.float),('psf_e2',np.float),
                                               ('logL_e1',np.float),('logL_e2',np.float),
                                               ('field_id',np.int),('good',np.bool)])
    results['g1_est'] = np.array(est1)
    results['g2_est'] = np.array(est2)
    results['g1_err'] = np.hstack(est1_err)
    results['g2_err'] = np.hstack(est2_err)
    results['g1_raw'] = np.array([np.mean(catalog['g1'])/(2*(1-np.var(catalog['g1']))) for catalog in catalogs])
    results['g2_raw'] = np.array([np.mean(catalog['g2'])/(2*(1-np.var(catalog['g2']))) for catalog in catalogs])
    results['psf_e1'] = np.array([catalog[0]['psf_e1'] for catalog in catalogs])
    results['psf_e2'] = np.array([catalog[0]['psf_e2'] for catalog in catalogs])
    results['logL_e1'] = np.array(logL_e1)
    results['logL_e2'] = np.array(logL_e2)
    results['field_id'] = np.array([catalog[0]['id'] / 1000000 for catalog in catalogs])

    truthTable = np.loadtxt(truthFile, dtype = [('field_id',np.int), ('g1',np.double), ('g2',np.double ) ])
    for i,this_result in enumerate(results):
        use = truthTable['field_id'] == this_result['field_id']
        if np.sum(use) == 1:
            results[i]['g1_true'] = truthTable[use]['g1']
            results[i]['g2_true'] = truthTable[use]['g2']
            results[i]['good'] = True
        else:
            results['good'] = False
    
    return results

def shear_model(x, m, a, c):
    # x should be 3 x N, where 0=gtrue, 1=epsf, 2=const
    return m*x[0,:] + a*x[1,:] + c*x[2,:]

def getCalibCoeff(results):
    from scipy.optimize import curve_fit
    A1 = np.column_stack([results['g1_true'], results['psf_e1'], np.ones_like(results['psf_e1'])]).transpose()
    B1 = results['g1_est'] - results['g1_true']
    ret1_val, covar1 = curve_fit(shear_model, A1, B1, sigma=results['g1_err'])
    A2 = np.column_stack([results['g2_true'], results['psf_e2'], np.ones_like(results['psf_e2'])]).transpose()
    B2 = results['g2_est'] - results['g2_true']
    ret2_val, covar2 = curve_fit(shear_model, A2, B2, sigma=results['g1_err'])
    
    coeff1 = {'m':ret1_val[0],
             'm_err':np.sqrt(covar1[0][0]),
             'a':ret1_val[1],
             'a_err':np.sqrt(covar1[1][1]),
             'c':ret1_val[2],
             'c_err':np.sqrt(covar1[2][2])}

    coeff2 = {'m':ret2_val[0],
             'm_err':np.sqrt(covar2[0][0]),
             'a':ret2_val[1],
             'a_err':np.sqrt(covar2[1][1]),
             'c':ret2_val[2],
             'c_err':np.sqrt(covar2[2][2])}
                 
    return coeff1,coeff2



def analyze(results_all,mc_type = None, logLcut1 = 3.6, logLcut2 = 3.6, clip=False):

    # use only the results with matching truth table entries.
    results = results_all[results_all['good'] & (results_all['logL_e1'] > logLcut1) & (results_all['logL_e2'] > logLcut2) ]
    res_exc = results_all[(results_all['logL_e1'] < logLcut1) & (results_all['logL_e2'] < logLcut2) ]
    # Apply a 10% outlier clipping:
    clipfrac = 2
    clip_interval1 = np.abs(np.percentile(results['g1_est'] - results['g1_true'],clipfrac/2.) -
                           np.percentile(results['g1_est'] - results['g1_true'],100-clipfrac/2.))
    keep1 = (np.abs(results['g1_est'] - results['g1_true'] ) < clip_interval1)
    clip_interval2 = np.abs(np.percentile(results['g2_est'] - results['g2_true'],clipfrac/2.) -
                           np.percentile(results['g2_est'] - results['g2_true'],100-clipfrac/2.))
    keep2 = (np.abs(results['g2_est'] - results['g2_true'] ) < clip_interval2) 
    keep = keep1 & keep2
    resclip = results[keep]
    
    figName = mc_type+'-simple'
    resraw = results_all.copy()
    resraw['g1_est'] = resraw['g1_raw']
    resraw['g2_est'] = resraw['g2_raw']
    
    coeff1_raw, coeff2_raw = getCalibCoeff(resraw)
    coeff1,coeff2 = getCalibCoeff(results)
    coeff_clipped1, coeff_clipped2 = getCalibCoeff(resclip)
    
    print ('Found coeff:\n m1 = %.4f +/- %.4f \n c1 = %.4f +/- %.4f \n a1 = %.4f +/- %.4f'
           %(coeff1['m'],coeff1['m_err'],coeff1['c'],coeff1['c_err'],coeff1['a'],coeff1['a_err']) )
    print ('Found coeff:\n m2 = %.4f +/- %.4f \n c2 = %.4f +/- %.4f \n a2 = %.4f +/- %.4f'
           %(coeff2['m'],coeff2['m_err'],coeff2['c'],coeff2['c_err'],coeff2['a'],coeff2['a_err']) )

    print "After "+str(clipfrac)+"% residual clipping, we get:"
    print ('Found coeff (clipped ):\n m1 = %.4f  c1 = %.4f \n m2 = %.4f  c2 = %.4f '
           %(coeff_clipped1['m'],coeff_clipped1['c'],coeff_clipped2['m'],coeff_clipped2['c']) )


    
    fig,((ax1,ax2),(ax3,ax4),(ax5,ax6),(ax7,ax8)) = plt.subplots(nrows=4,ncols=2,figsize=(14,21))
    xlim = [-0.06, 0.06]
    y_interval_raw = np.abs(np.percentile(np.hstack((results['g1_raw']-results['g1_true'],results['g2_raw']-results['g2_true'])),10) -
                            np.percentile(np.hstack((results['g1_raw']-results['g1_true'],results['g2_raw']-results['g2_true'])),90))
    y_interval_est = np.abs(np.percentile(np.hstack((results['g1_est']-results['g1_true'],results['g2_est']-results['g2_true'])),10) -
                            np.percentile(np.hstack((results['g1_est']-results['g1_true'],results['g2_est']-results['g2_true'])),90))

    ylim_raw = [-3*y_interval_raw, 3*y_interval_raw]
    ylim_est = [-3*y_interval_est, 3*y_interval_est]
    #--------------------------------------------------
    # First, the raw results.
    #--------------------------------------------------
    ax1.plot(results['g1_true'],results['g1_raw'] - results['g1_true'],'.',color='blue')
    ax1.plot([-1,1],[coeff1_raw['c'] -coeff1_raw['m'], coeff1_raw['c'] + coeff1_raw['m']],color='cyan')
    ax1.axhline(0,linestyle='--',color='red')
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim_raw)
    ax1.annotate('m = %.4f +/- %.4f \n a = %.4f +/- %.4f \n c = %.4f +/0 %.4f'%
                 (coeff1_raw['m'],coeff1_raw['m_err'],coeff1_raw['a'],coeff1_raw['a_err'],
                  coeff1_raw['c'],coeff1_raw['c_err']),xy=(0.05, 0.85), xycoords='axes fraction')

    
    ax2.plot(results['g2_true'],results['g2_raw'] - results['g2_true'],'.',color='blue')
    ax2.plot([-1,1],[coeff2_raw['c'] -coeff2_raw['m'], coeff2_raw['c'] + coeff2_raw['m']],color='cyan')
    ax2.axhline(0,linestyle='--',color='red')
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim_raw)
    ax2.annotate('m = %.4f +/- %.4f \n a = %.4f +/- %.4f \n c = %.4f +/0 %.4f'%
                 (coeff2_raw['m'],coeff2_raw['m_err'],coeff2_raw['a'],coeff2_raw['a_err'],
                  coeff2_raw['c'],coeff2_raw['c_err']),xy=(0.05, 0.85), xycoords='axes fraction')
    
    #--------------------------------------------------    
    # Next, the MC'd results.
    #--------------------------------------------------
    ax3.plot(results['g1_true'],results['g1_est'] - results['g1_true'],'.',color='blue')
    ax3.plot(res_exc['g1_true'],res_exc['g1_est'] - res_exc['g1_true'],'s',color='red')
    ax3.plot([-1,1],[coeff1['c'] -coeff1['m'], coeff1['c'] + coeff1['m']],color='orange')
    ax3.plot([-1,1],[coeff_clipped1['c'] -coeff_clipped1['m'], coeff_clipped1['c'] + coeff_clipped1['m']],color='cyan')
    ax3.axhline(0,linestyle='--',color='red')
    ax3.set_xlim(xlim)
    ax3.set_ylim(ylim_est)
    ax3.annotate('m = %.4f +/- %.4f \n a = %.4f +/- %.4f \n c = %.4f +/0 %.4f'%
                 (coeff1['m'],coeff1['m_err'],coeff1['a'],coeff1['a_err'],
                  coeff1['c'],coeff1['c_err']),xy=(0.05, 0.85), xycoords='axes fraction')

        
    ax4.plot(results['g2_true'],results['g2_est'] - results['g2_true'],'.',color='blue')
    ax4.plot(res_exc['g2_true'],res_exc['g2_est'] - res_exc['g2_true'],'s',color='red')
    ax4.plot([-1,1],[coeff2['c'] -coeff2['m'], coeff2['c'] + coeff2['m']],color='orange')
    ax4.plot([-1,1],[coeff_clipped2['c'] -coeff_clipped2['m'], coeff_clipped2['c'] + coeff_clipped2['m']],color='cyan')
    ax4.axhline(0,linestyle='--',color='red')
    ax4.set_xlim(xlim)
    ax4.set_ylim(ylim_est)
    ax4.annotate('m = %.4f +/- %.4f \n a = %.4f +/- %.4f \n c = %.4f +/0 %.4f'%
                 (coeff2['m'],coeff2['m_err'],coeff2['a'],coeff2['a_err'],
                  coeff2['c'],coeff2['c_err']),xy=(0.05, 0.85), xycoords='axes fraction')

    ax5.plot(results_all['logL_e1'],results_all['g1_est']-results_all['g1_true'],'.')
    ax5.set_xlabel('logL')
    ax5.axhline(0,color='red',linestyle='--')
    ax5.axvline(logLcut1,color='red')
    ax6.plot(results_all['logL_e2'],results_all['g2_est']-results_all['g2_true'],'.')
    ax6.set_xlabel('logL')
    ax6.axvline(logLcut2,color='red')
    ax6.axhline(0,color='red',linestyle='--')

    ax7.plot(results['psf_e1'],results['g1_est']-results['g1_true'],'.',color='blue')
    ax7.plot(res_exc['psf_e1'],res_exc['g1_est']-res_exc['g1_true'],'s',color='red')
    ax7.plot(results['psf_e1'],results['g1_est']-results['g1_true'],'.',color='blue')
    ax7.plot(res_exc['psf_e1'],res_exc['g1_est']-res_exc['g1_true'],'s',color='red')
    ax7.plot([-1,1],[coeff1['c'] - coeff1['a'], coeff1['c'] + coeff1['a']],color='orange')
    ax7.plot([-1,1],[coeff_clipped1['c'] - coeff_clipped1['a'], coeff_clipped1['c'] + coeff_clipped1['a']],color='orange')
    ax7.axhline(0,linestyle='--',color='red')
    ax7.set_xlim(-0.2,0.2)
    ax7.set_xlabel('psf_e1')
        
    ax8.plot(results['psf_e2'],results['g2_est']-results['g2_true'],'.',color='blue')
    ax8.plot(res_exc['psf_e2'],res_exc['g2_est']-res_exc['g2_true'],'s',color='red')
    ax8.plot([-1,1],[coeff2['c'] - coeff2['a'], coeff2['c'] + coeff2['a']],color='orange')
    ax8.plot([-1,1],[coeff_clipped2['c'] - coeff_clipped2['a'], coeff_clipped2['c'] + coeff_clipped2['a']],color='orange')
    ax8.axhline(0,linestyle='--',color='red')
    ax8.set_xlim(-0.2,0.2)
    ax8.set_xlabel('psf_e2')
    
        
    fig.savefig(figName)
    if clip is False:
        return coeff1, coeff2, coeff1_raw, coeff2_raw
    else:
        return coeff_clipped1, coeff_clipped2, coeff1_raw, coeff2_raw

def determineLoglCuts(catalog, percentile = None):
    if percentile is not None:
        logL1cut = np.percentile(catalog['logL_e1'],percentile)*1.001
        logL2cut = np.percentile(catalog['logL_e2'],percentile)*1.001
    return logL1cut, logL2cut
    
def main(argv):

    import argparse

    description = """Analyze MetaCalibration outputs from Great3 and Great3++ simulations."""
    mc_choices =['regauss', 'regauss-sym', 'ksb', 'none-regauss', 'moments', 'noaber-regauss-sym','noaber-regauss','rgc-regauss','rgc-noaber-regauss','rgc-fixedaber-regauss', 'rgc-ksb','cgc-noaber-precise']
    # Note: The above line needs to be consistent with the choices in getAllCatalogs.

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--path", dest="path", type=str, default="../Great3/",
                        help="path to MetaCalibration output catalogs")
    parser.add_argument("-mc","--mc_type", dest="mc_type", type=str, default="regauss",
                        choices = mc_choices, help="metcalibration catalog type to use")
    parser.add_argument("-o", "--outfile", dest = "outfile", type = str, default = "tmp_outfile.txt",
                        help = "destination for output per-field shear catalogs.")
    parser.add_argument("-dp", "--doplot", dest = "doplot", action="store_true")
    parser.add_argument("-p", "--percentile_cut", dest="percentile_cut",
                        help="percentile",type= float, default = 0)
    parser.add_argument("-a", "--do_all", dest = "do_all", action="store_true", default = False)
    parser.add_argument("-c", "--clip", dest = "clip", action='store_true', help="clip large residuals", default = False)
    parser.add_argument("-sn", "--snos_cut", dest="sn_cut",
                        help="signal-to-noise cut",type= float, default = 0)

    args = parser.parse_args(argv[1:])
    if args.sn_cut > 0:
        sn_cut = args.sn_cut
    else:
        sn_cut = None
        path = args.path


    outfile = args.outfile
    catalogs, truthfile = getAllCatalogs()
    results = shear_est(catalogs,truthfile, mc_type = args.mc_type)
    logLcut1, logLcut2 = determineLoglCuts(results, percentile = args.percentile_cut)
    coeff1, coeff2,_,_ = analyze(results,mc_type= args.mc_type, logLcut1 = logLcut1, logLcut2 = logLcut2)

    

            
if __name__ == "__main__":
    import pdb, traceback
    try:
        main(sys.argv)
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
