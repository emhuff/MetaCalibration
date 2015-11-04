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
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import em_student_t as emt
    
def getAllCatalogs( path = '../Great3/', mc_type = None, sn_cut = None ):

    globalPath = path
    if mc_type=='regauss':
        path = path+'Outputs-Regauss-BugFix/output_catalog*.fits'
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
        path = path+'Outputs-Regauss-NoAber-Bugfix/cgc_noaber_metacalfix_regauss*.fits'
        truthFile = 'cgc-noaber-truthtable.txt'
    elif mc_type=='cgc-noaber-precise':
        path = path+'Outputs-Regauss-NoAber-HighPrec/cgc_noaber_precise3_metacal*.fits'
        truthFile = 'cgc-noaber-truthtable.txt'
    elif mc_type == 'rgc-regauss':
        path = path+'Outputs-Real-Regauss-BugFix/output_catalog*.fits'
        truthFile = 'rgc-truthtable.txt'
    elif mc_type == 'rgc-noaber-regauss':
        path = path+'Outputs-Real-NoAber-Regauss-BugFix/rgc_noaber_metacalfix_regauss*.fits'
        truthFile = 'rgc-noaber-truthtable.txt'
    elif mc_type == 'rgc-ksb':
        path= path+'Outputs-Real-KSB/output_catalog*.fits'
        truthFile = 'rgc-truthtable.txt'
    elif mc_type=='rgc-fixedaber-regauss':
        path = path+'Outputs-Real-Regauss-FixedAber-BugFix/rgc_fixedaber_metacalfix_regauss*.fits'
        truthFile = 'rgc-fixedaber-truthtable.txt'

    else:
        raise RuntimeError('Unrecognized mc_type: %s'%mc_type)

    if sn_cut is not None:
        truthPath = globalPath+'Truth/'
        if mc_type == 'rgc-noaber-regauss':
            truthPath = truthPath+'rgc-noaber/'
        if (mc_type == 'noaber-regauss') or (mc_type == 'cgc-noaber-precise'):
            truthPath = truthPath+'cgc-noaber/'
        if (mc_type == 'regauss') or (mc_type == 'regauss-bugfix'):
            truthPath = truthPath+'cgc/'
            
    catFiles = glob.glob(path)
    if len(catFiles) == 0:
        raise RuntimeError("No catalogs found with path %s!"%path)
    catalogs = []
    for thisFile in catFiles:
        this_catalog = fits.getdata(thisFile)
        keep  =   (this_catalog['g1'] != -10) & (this_catalog['g2'] != -10) & (this_catalog['weight'] > 0)
        this_catalog = this_catalog[keep]
        if (mc_type=='moments') or (mc_type=='ksb'):
            this_catalog['a1'] = this_catalog['a1']/2.
            this_catalog['a2'] = this_catalog['a2']/2.
            
        else:
            this_catalog = fits.getdata(thisFile)
            keep  =   (this_catalog['g1'] != -10) & (this_catalog['g2'] != -10) & (this_catalog['weight'] > 0) 
            this_catalog = this_catalog[keep]
        if sn_cut is not None:
            # Parse thisFile to figure out where the truth catalog
            # lives.
            pattern = re.compile("-(\d*).fits")
            thisField = pattern.findall(thisFile)[0]
            thisTruthFile = truthPath + 'subfield_catalog-'+thisField+'.fits'
            truthCat = fits.getdata(thisTruthFile)
            keep  =   (truthCat['gal_sn'] > (sn_cut) )
            use = np.in1d(this_catalog['id'],truthCat[keep]['id'])
            this_catalog = this_catalog[use]
            truthCat = truthCat[use]

        catalogs.append(this_catalog)
    return catalogs, truthFile

def reconstructMetacalMeas(g=None, R=None, a = None, c=None, psf_e=None, delta_g = 0.01 ):
    esum = 2*(c + g)
    ediff = 2*delta_g * R
    ep = (esum + ediff)/2. - a * psf_e
    em = (esum - ediff)/2. - a * psf_e
    return ep,em

def bootstrap_err_est(catalog=None, g1Tag='g1',g2Tag='g2', method=None, n_iter = 100):
    
    e1 = []
    e2 = []
    for i in xrange(n_iter):
        this_catalog = np.random.choice(catalog,size=catalog.size,replace=True)
        this_e1 = method(this_catalog[g1Tag])
        this_e2 = method(this_catalog[g2Tag])
        e1.append(this_e1)
        e2.append(this_e2)
    e1 = np.array(e1)
    e2 = np.array(e2)
    #e1_err = np.abs(np.percentile(e1,36) - np.percentile(e1,64))
    #e2_err = np.abs(np.percentile(e2,36) - np.percentile(e2,64))
    e1_err = np.std(e1)
    e2_err = np.std(e2)
    return e1_err, e2_err

def shear_avg(e, weights = True, n=2, exp=False, scale = 0.03):
    if weights is None:
        return np.mean(e)
    else:
        if exp is False:
            weight = 1./(scale**n + np.abs(e)**n)
        else:
            weight = np.exp(- 0.5*np.abs(e/scale)**n)
        for i in xrange(25):
            eavg = np.average(e,weights=weight)
            if exp is False:
                weight = 1./(scale**n + np.abs(e-eavg)**n)
            else:
                weight = np.exp(- 0.5*((e-eavg)/scale)**n)
        return np.average(e,weights=weight), np.average(np.log(weight/np.sum(weight)))

def shear_em(e):
    mu, sigma, nu = emt.estimate_t(e)
    return mu, sigma, nu



def shear_est(catalogs, truthFile, delta_g = 0.01, weights = True):

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
    bins = np.linspace(-10,10,250)
    xx = (bins[0:-1] + bins[1:])/2.
    yy_est =  emt.t(xx, mu1, sigma1, nu1)
    yy_est = yy_est/np.sum(yy_est)
    fig,ax = plt.subplots()
    yy_data,b = np.histogram(e1_master,bins=bins)
    ax.plot(xx,yy_est,label='model')
    ax.plot(xx,yy_data*1./np.sum(yy_data),label='data')
    plt.legend(loc='best')
    plt.yscale('log')
    fig.savefig('prior_model_comparison')


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


        g1p, _ = shear_avg(e1p)
        g10, logL1 = shear_avg(e10)
        g1m, _ = shear_avg(e1m)

        g2p, _ = shear_avg(e2p)
        g20, logL2 = shear_avg(e20)
        g2m, _ = shear_avg(e2m)

        m1 = (g1p - g1m)/(2*delta_g)
        c1 = (g1p + g1m)/2. - g10 
        m2 = (g2p - g2m)/(2*delta_g)
        c2 = (g2p + g2m)/2. - g20
        
        est1.append((g10 - c1)/m1)
        est2.append((g20 - c2)/m2)

        logL_e1.append(logL1)
        logL_e2.append(logL2)
    
        this_err1 = np.std(catalog['g1'])/np.sqrt(catalog.size)
        this_err2 = np.std(catalog['g2'])/np.sqrt(catalog.size)

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
    results['g1_err'] = np.array(est1_err)
    results['g2_err'] = np.array(est2_err)
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
    # report results
    use = results['good']
    w1 = w=1./results[use]['g1_err']**2
    w1[~np.isfinite(w1)]=0.
    w2 = w=1./results[use]['g2_err']**2
    w2[~np.isfinite(w1)]=0.
    
    coeff_all1,cov1 = np.polyfit(results[use]['g1_true'],results[use]['g1_est']-results[use]['g1_true'],1,
                            cov=True,w=w1)
    coeff_all2,cov2 = np.polyfit(results[use]['g2_true'],results[use]['g2_est']-results[use]['g2_true'],1,
                            cov=True,w=w2)


    #print 'Found coeff:\n m1 = %.4f +/- %.4f \n c1 = %.4f +/- %.4f'%(coeff_all1[0],cov1[0,0],coeff_all1[1],cov1[1,1])
    #print 'Found coeff:\n m2 = %.4f +/- %.4f \n c2 = %.4f +/- %.4f'%(coeff_all2[0],cov2[0,0],coeff_all2[1],cov2[1,1])

    # Refit after clipping bad coefficients.
    res = results[use]
    clipfrac = 5
    clip_interval = np.abs(np.percentile(res['g1_est'] - res['g1_true'],clipfrac/2.) -
                           np.percentile(res['g1_est'] - res['g1_true'],100-clipfrac/2.))
    keep = np.abs(res['g1_est'] - res['g1_true'] ) < clip_interval
    resclip = res[keep]
    coeff_clipped1,cov_clipped1 = np.polyfit(resclip['g1_true'],resclip['g1_est']-resclip['g1_true'],1,
                                cov=True,w=1./resclip['g1_err']**2)
    coeff_clipped2,cov_clipped2 = np.polyfit(resclip['g2_true'],resclip['g2_est']-resclip['g2_true'],1,
                                cov=True,w=1./resclip['g2_err']**2)

    #print "After "+str(clipfrac)+"% residual clipping, we get:"
    #print 'Found coeff:\n m1 = %.4f +/- %.4f \n c1 = %.4f +/- %.4f'%(coeff_clipped1[0],cov_clipped1[0,0],coeff_clipped1[1],cov_clipped1[1,1])
    #print 'Found coeff:\n m2 = %.4f +/- %.4f \n c2 = %.4f +/- %.4f'%(coeff_clipped2[0],cov_clipped2[0,0],coeff_clipped2[1],cov_clipped2[1,1])
    
    return results


def makePlots(results_all,mc_type = None, logLcut1 = 3.6, logLcut2 = 3.6):

    # use only the results with matching truth table entries.
    results = results_all[results_all['good'] & (results_all['logL_e1'] > logLcut1) & (results_all['logL_e2'] > logLcut2) ]
    res_exc = results_all[(results_all['logL_e1'] < logLcut1) & (results_all['logL_e2'] < logLcut2) ]
    # Apply a 10% outlier clipping:
    clipfrac = 5
    clip_interval = np.abs(np.percentile(results['g1_est'] - results['g1_true'],clipfrac/2.) -
                           np.percentile(results['g1_est'] - results['g1_true'],100-clipfrac/2.))
    keep = np.abs(results['g1_est'] - results['g1_true'] ) < clip_interval
    resclip = results[keep]
    coeff_clipped1 = np.polyfit(resclip['g1_true'],resclip['g1_est']-resclip['g1_true'],1)
    coeff_clipped2 = np.polyfit(resclip['g2_true'],resclip['g2_est']-resclip['g2_true'],1)

    
    figName = mc_type+'-simple'
    coeff1 = np.polyfit(results['g1_true'],results['g1_est']-results['g1_true'],1)
    coeff2 = np.polyfit(results['g2_true'],results['g2_est']-results['g2_true'],1)
    coeff1_raw = np.polyfit(results['g1_true'],results['g1_raw']-results['g1_true'],1)
    coeff2_raw = np.polyfit(results['g2_true'],results['g2_raw']-results['g2_true'],1)
    print 'Found coeff:\n m1 = %.4f  c1 = %.4f \n m2 = %.4f  c2 = %.4f '%(coeff1[0],coeff1[1],coeff2[0],coeff2[1])
    print "After "+str(clipfrac)+"% residual clipping, we get:"
    print 'Found coeff (clipped ):\n m1 = %.4f  c1 = %.4f \n m2 = %.4f  c2 = %.4f '%(coeff_clipped1[0],coeff_clipped1[1],coeff_clipped2[0],coeff_clipped2[1])


    
    fig,((ax1,ax2),(ax3,ax4),(ax5,ax6)) = plt.subplots(nrows=3,ncols=2,figsize=(14,21))
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
    ax1.plot([-1,1],[coeff1_raw[1] -coeff1_raw[0], coeff1_raw[1] + coeff1_raw[0]],color='cyan')
    ax1.axhline(0,linestyle='--',color='red')
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim_raw)
    
    ax2.plot(results['g2_true'],results['g2_raw'] - results['g2_true'],'.',color='blue')
    ax2.plot([-1,1],[coeff2_raw[1] -coeff2_raw[0], coeff2_raw[1] + coeff2_raw[0]],color='cyan')
    ax2.axhline(0,linestyle='--',color='red')
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim_raw)
    
    #--------------------------------------------------    
    # Next, the MC'd results.
    #--------------------------------------------------
    ax3.plot(results['g1_true'],results['g1_est'] - results['g1_true'],'.',color='blue')
    ax3.plot(res_exc['g1_true'],res_exc['g1_est'] - res_exc['g1_true'],'.',color='red')
    ax3.plot([-1,1],[coeff1[1] -coeff1[0], coeff1[1] + coeff1[0]],color='orange')
    ax3.plot([-1,1],[coeff_clipped1[1] -coeff_clipped1[0], coeff_clipped1[1] + coeff_clipped1[0]],color='cyan')
    ax3.axhline(0,linestyle='--',color='red')
    ax3.set_xlim(xlim)
    ax3.set_ylim(ylim_est)
    
    ax4.plot(results['g2_true'],results['g2_est'] - results['g2_true'],'.',color='blue')
    ax4.plot(res_exc['g2_true'],res_exc['g2_est'] - res_exc['g2_true'],'.',color='red')
    ax4.plot([-1,1],[coeff2[1] -coeff2[0], coeff2[1] + coeff2[0]],color='orange')
    ax4.plot([-1,1],[coeff_clipped2[1] -coeff_clipped2[0], coeff_clipped2[1] + coeff_clipped2[0]],color='cyan')
    ax4.axhline(0,linestyle='--',color='red')
    ax4.set_xlim(xlim)
    ax4.set_ylim(ylim_est)

    ax5.plot(results_all['logL_e1'],results_all['g1_est']-results_all['g1_true'],'.')
    ax5.set_xlabel('logL')
    ax5.axhline(0,color='red',linestyle='--')
    ax5.axvline(logLcut1,color='red')
    ax6.plot(results_all['logL_e2'],results_all['g2_est']-results_all['g2_true'],'.')
    ax6.set_xlabel('logL')
    ax6.axvline(logLcut2,color='red')
    ax6.axhline(0,color='red',linestyle='--')
    
    fig.savefig(figName)

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
                        help="percentile",type= float, default = 10)
    parser.add_argument("-a", "--do_all", dest = "do_all", action="store_true", default = False)
    parser.add_argument("-sn", "--snos_cut", dest="sn_cut",
                        help="signal-to-noise cut",type= float, default = 0)

    args = parser.parse_args(argv[1:])
    if args.sn_cut > 0:
        sn_cut = args.sn_cut
    else:
        sn_cut = None
        path = args.path

    outfile = args.outfile
    print 'Getting catalogs from path %s and mc_type %s'%(path, args.mc_type)
    catalogs, truthfile = getAllCatalogs(path=path, mc_type=args.mc_type,sn_cut = sn_cut)
    results = shear_est(catalogs,truthfile)
    logLcut1, logLcut2 = determineLoglCuts(results, percentile = args.percentile_cut)
    makePlots(results,mc_type= args.mc_type, logLcut1 = logLcut1, logLcut2 = logLcut2)

       
if __name__ == "__main__":
    import pdb, traceback
    try:
        main(sys.argv)
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
