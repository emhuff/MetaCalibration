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
    total_number = 0
    excluded_number = 0
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



def shear_est(catalogs, truthTable, delta_g = 0.01, weights = True,mc_type=None):

    est1 = []
    est2 = []
    est1_err = []
    est2_err = []
    logL_e1 = []
    logL_e2 = []

    e1_master = np.hstack([catalog['g1'] - catalog['c1'] - catalog['a1']*catalog['psf_e1'] for catalog in catalogs])
    
    catalogs_all = np.hstack(catalogs)
    
    e1p_master, e1m_master = reconstructMetacalMeas(g=catalogs_all['g1'], R=catalogs_all['R1'],
                                                    a = catalogs_all['a1'], c=catalogs_all['c1'],
                                                    psf_e=catalogs_all['psf_e1'], delta_g = delta_g )
    e2_master = np.hstack([catalog['g2'] - catalog['c2'] - catalog['a2']*catalog['psf_e2'] for catalog in catalogs])
    e2p_master, e2m_master = reconstructMetacalMeas(g=catalogs_all['g2'], R=catalogs_all['R2'],
                                                    a = catalogs_all['a2'], c=catalogs_all['c2'],
                                                    psf_e=catalogs_all['psf_e2'], delta_g = delta_g )

    mu1, sigma1, nu1 = shear_em(e1_master)
    mu1p,sigma1p,nu1p = shear_em(e1p_master)
    mu1m,sigma1m,nu1m = shear_em(e1m_master)
    m1,c1 =np.polyfit([-delta_g,0.,delta_g],[mu1m,mu1,mu1p],1)
    s1_2,s1_1,s1_0 = np.polyfit([mu1m,mu1,mu1p],[sigma1m,sigma1,sigma1p],2)
    n1_2,n1_1,n1_0 = np.polyfit([mu1m,mu1,mu1p],[nu1m,nu1,nu1p],2)
    
    mu2, sigma2, nu2 = shear_em(e2_master)
    mu2p,sigma2p,nu2p = shear_em(e2p_master)
    mu2m,sigma2m,nu2m = shear_em(e2m_master)
    m2,c2 =np.polyfit([-delta_g,0.,delta_g],[mu2m,mu2,mu2p],1)
    s2_2,s2_1,s2_0 = np.polyfit([mu2m,mu2,mu2p],[sigma2m,sigma2,sigma2p],2)
    n2_2,n2_1,n2_0 = np.polyfit([nu1m,nu1,nu1p],[nu1m,nu1,nu1p],2)
        
    plt.plot([-delta_g,0.,delta_g],[nu1m, nu1,nu1p],label='e1')
    plt.plot([-delta_g,0.,delta_g],[nu2m, nu2,nu2p],label='e2')
    plt.legend(loc='best')
    plt.show()

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
        mu1p,sigma1p,nu1p= shear_em(e1p)
        mu1m,sigma1m,nu1m= shear_em(e1m)
        
        mu2, sigma2, nu2 = shear_em(e20)
        mu2p,sigma2p,nu2p= shear_em(e2p)
        mu2m,sigma2m,nu2m= shear_em(e2m)
        
        _, logL1 = shear_avg(e10,n=nu1_global, scale = sigma1_global)
        _, logL2 = shear_avg(e20,n=nu2_global, scale = sigma2_global)

        this_sigma1 = s1_2*mu1**2 + s1_1*mu1 + s1_0
        this_nu1 = n1_2*mu1**2 + n1_1 * mu1 + n1_0
        this_sigma2 = s2_2*mu2**2 + s2_1*mu2 + s2_0
        this_nu2 = n2_2*mu1**2 + n2_1 * mu2 + n2_0
                
        g1p, _ = shear_avg(e1p,n=this_nu1, scale = this_sigma1)
        g10, _ = shear_avg(e10,n=this_nu1, scale = this_sigma1)
        g1m, _ = shear_avg(e1m,n=this_nu1, scale = this_sigma1)

        g2p, _ = shear_avg(e2p,n=this_nu2, scale = this_sigma2)
        g20, _ = shear_avg(e20,n=this_nu2, scale = this_sigma2)
        g2m, _ = shear_avg(e2m,n=this_nu2, scale = this_sigma2)

        
        #m1 = (g1p - g1m)/(2*delta_g)
        #c1 = (g1p + g1m)/2. - g10 
        #m2 = (g2p - g2m)/(2*delta_g)
        #c2 = (g2p + g2m)/2. - g20

        est1.append((g10 - c1)/m1)
        est2.append((g20 - c2)/m2)

        logL_e1.append(logL1)
        logL_e2.append(logL2)


            
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
                                               ('field_id',np.int)])
      shears_arr = np.array(shears)
      shears_obj['g1'] = shears_arr[:,0]
      shears_obj['g2'] = shears_arr[:,1]
      shears_obj['field_id'] = np.arange(len(shears),dtype=np.int)
      return shears_obj

def doPlots(data,outfile = None):
    truthTable = get_truthtable()

    coeff1, covar1 = np.polyfit(truthTable['g1'],data['g1_est'] - truthTable['g1'],1,cov=True)
    coeff2, covar2 = np.polyfit(truthTable['g2'],data['g2_est'] - truthTable['g2'],1,cov=True)
    print 'm1 = '+str(coeff1[0])+'+/- '+str(np.sqrt(covar1[0,0]))+', c1 = '+str(coeff1[1])+'  '+str(np.sqrt(covar1[1,1]))
    print 'm2 = '+str(coeff2[0])+'+/- '+str(np.sqrt(covar2[0,0]))+', c2 = '+str(coeff2[1])+'  '+str(np.sqrt(covar2[1,1]))
    fig,((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(nrows=3,ncols=2,figsize=(14,7))
    ax1.plot(truthTable['g1'],data['g1_est'] - truthTable['g1'],'.')
    ax1.axhline(0,linestyle='--',color='red')
    ax1.plot(truthTable['g1'],coeff1[0]*truthTable['g1'] + coeff1[1],color='cyan')
    ax1.set_ylim(-0.02,0.02)
    ax2.plot(truthTable['g2'],data['g2_est'] - truthTable['g2'],'.')
    ax2.plot(truthTable['g2'],coeff2[0]*truthTable['g2'] + coeff2[1],color='cyan')
    ax2.axhline(0,linestyle='--',color='red')
    ax2.set_ylim(-0.02,0.02)

    ax3.plot(data['logL_e1'],data['g1_est'] - truthTable['g1'],'.')
    ax3.set_ylim(-0.02,0.02)
    ax3.axhline(0,linestyle='--',color='red')
    ax4.plot(data['logL_e2'],data['g2_est'] - truthTable['g2'],'.')
    ax4.set_ylim(-0.02,0.02)
    ax4.axhline(0,linestyle='--',color='red')
    
    ax5.plot(data['psf_e1'],data['g1_est'] - truthTable['g1'],'.')
    ax5.set_ylim(-0.02,0.02)
    ax5.axhline(0,linestyle='--',color='red')
    
    ax6.plot(data['psf_e2'],data['g2_est'] - truthTable['g2'],'.')
    ax6.axhline(0,linestyle='--',color='red')
    ax6.set_ylim(-0.02,0.02)

    fig.savefig(outfile)
    pass

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
    catalogs = getAllCatalogs()
    truthTable = get_truthtable()
    results = shear_est(catalogs,truthTable, mc_type = args.mc_type)
    logLcut1, logLcut2 = determineLoglCuts(results, percentile = args.percentile_cut)
    doPlots(results,outfile = 'est_simple')

    

            
if __name__ == "__main__":
    import pdb, traceback
    try:
        main(sys.argv)
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
