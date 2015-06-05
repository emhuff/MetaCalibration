#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
from scipy.optimize import curve_fit
import sys
import socket



def get_true_mean_shear(mc_type =  None):
    all_branches = ['regauss', 'regauss-sym', 'ksb', 'none-regauss', 'moments',
                    'noaber-regauss-sym', 'noaber-regauss','rgc-regauss',
                    'rgc-noaber-regauss','rgc-fixedaber-regauss']
    if mc_type not in all_branches:
        print "mc_type must be one of: "+' '.join(all_branches)
        raise Exception(mc_type+' is not a legitimate mc_type')

    if mc_type in ['regauss','regauss-sym','ksb','moments']:
        # cgc from GREAT3
        g1 =-6.97501411e-04
        g2 = 2.72438059e-03
    if mc_type in ['rgc-regauss']:
        # rgc from GREAT3
        g1 =  -0.0027895457500000005
        g2 = 0.0015939298500000005
    if mc_type in ['rgc-noaber-regauss','noaber-regauss-sym','noaber-regauss']:
        # both rgc and cgc with no aberrations.
        g1 = 0.00066805100000000002
        g2 = -0.00269821215
    if mc_type in ['rgc-fixedaber-regauss']:
        # rgc with large fixed aberrations
        g1 = -0.00113594965
        g2 =  0.00066726915
    return g1, g2


   
def main(argv):

    import argparse

    description = """Choose optimal nbins, likelihood cut for MetaCalibration on Great3, Great3++ """
    mc_choices =['regauss', 'regauss-sym', 'ksb', 'none-regauss', 'moments', 'noaber-regauss-sym', 'noaber-regauss','rgc-regauss','rgc-noaber-regauss','rgc-fixedaber-regauss']
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
    parser.add_argument("-c", "--clobber", dest="clobber", action="store_true")
    args = parser.parse_args(argv[1:])

    mc_type = args.mc_type
    
    filepref = 'outputs/output-'+mc_type
    filesuff = '.dat'
    if 'compute' in socket.gethostname() or 'coma' in socket.gethostname():
        rootdir = 'rachel_root_here'

    else:
        rootdir = '../Great3/'

    outpref = 'outputs/'+mc_type+'-'

    true_mean_g1,  true_mean_g2 = get_true_mean_shear( mc_type =  mc_type)
    

    n_bins = np.arange(30,151,12)
    percentile_vals = [0,0]#[0.1, 5] # in a case where there might be some outliers, set this to something like 0.1, 5
    n_logl_vals = 10

    
    mean_g1 = np.zeros((len(n_bins), n_logl_vals))
    mean_g2 = np.zeros((len(n_bins), n_logl_vals))
    sig_g1 = np.zeros((len(n_bins), n_logl_vals))
    sig_g2 = np.zeros((len(n_bins), n_logl_vals))

    for n_indx, n in enumerate(n_bins):
        # construct filename
        filename = '%s%d%s'%(filepref,n,filesuff)
        if (not os.path.exists(filename)) or (args.clobber):
            print 'Generating file %s'%filename
            tmp_command = 'python shear_ensemble_est.py --path %s --mc_type %s -o %s -n %d'%(rootdir,mc_type,filename,n)
            print tmp_command
            p = subprocess.Popen(tmp_command, shell=True, close_fds=True)
            p.wait()
            print 'Done running inference'
        # read in data
        print 'Reading from file ',filename
        dat = np.loadtxt(filename).transpose()
        field_id = dat[0,:]
        g1_opt = dat[3,:]
        g2_opt = dat[4,:]
        g1_var = dat[5,:]
        g2_var = dat[6,:]
        psf_e1 = dat[7,:]
        psf_e2 = dat[8,:]
        logl1 = dat[9,:]
        logl2 = dat[10,:]
        # If we haven't set up the logl cutoffs to use, then set it up now:
        if 'logl_cutoffs' not in locals():
            # concatenate the list of log likelihoods for both components
            logl = np.concatenate((logl1, logl2))
            if percentile_vals[0] > 0.:
                logl_cutoff_min = np.percentile(logl, percentile_vals[0])
            else:
                logl_cutoff_min = -1.e7
            if percentile_vals[1] > 0.:
                logl_cutoff_max = np.percentile(logl, percentile_vals[1])
            else:
                logl_cutoff_max = -50
            #logl_cutoffs = np.linspace(logl_cutoff_min, logl_cutoff_max, n_logl_vals)
            logl_cutoffs = -np.logspace(np.log10(np.abs(logl_cutoff_max)),np.log10(np.abs(logl_cutoff_min)),n_logl_vals)[::-1]
            
        # Loop over nbins, logL_cut, and make plots of the mean shear as a function of these two.
        for logl_indx, logl_cut in enumerate(logl_cutoffs):
            print "Log likelihood cutoff: ",logl_cut
            to_save_1 = logl1 > logl_cut
            use_g1_opt = g1_opt[to_save_1]
            use_g1_var = g1_var[to_save_1]
            use_psf_e1 = psf_e1[to_save_1]
    
            to_save_2 = logl2 > logl_cut
            use_g2_opt = g2_opt[to_save_2]
            use_g2_var = g2_var[to_save_2]
            use_psf_e2 = psf_e2[to_save_2]

            print "Using ",len(use_g1_opt),' and ',len(use_g2_opt),' for g1 and g2'


            # compute and store <gamma>
            mean_g1[n_indx][logl_indx]=np.mean(use_g1_opt)
            mean_g2[n_indx][logl_indx]=np.mean(use_g2_opt)
            print  "Found mean shears:",np.mean(use_g1_opt), np.mean(use_g2_opt)
            # store sigma_gamma
            sig_g1[n_indx][logl_indx]=np.mean(np.sqrt(use_g1_var))
            sig_g2[n_indx][logl_indx]=np.mean(np.sqrt(use_g2_var))

            # compute and store <gamma>
            mean_g1[n_indx][logl_indx]=np.mean(use_g1_opt)
            mean_g2[n_indx][logl_indx]=np.mean(use_g2_opt)

            # store sigma_gamma
            sig_g1[n_indx][logl_indx]=np.mean(np.sqrt(use_g1_var))
            sig_g2[n_indx][logl_indx]=np.mean(np.sqrt(use_g2_var))

        # just make plot of shear residual vs. n, logL
        mean_g1 -= true_mean_g1
        mean_g2 -= true_mean_g2
        fig = plt.figure()
        vmax = max(np.max(mean_g1[np.isfinite(mean_g1)]), -np.min(mean_g1[np.isfinite(mean_g1)]))
        print "Plotting with vmax =", vmax
        plt.imshow(mean_g1.transpose(), extent=(min(n_bins), max(n_bins), min(np.log10(logl_cutoffs)), max(np.log10(logl_cutoffs))),
                   interpolation='bicubic', aspect='auto', vmin=-vmax, vmax=vmax, cmap=plt.cm.bwr)
        plt.colorbar()
        plt.savefig(outpref+'mean_g1_2d.png')
        fig = plt.figure()
        plt.imshow(mean_g2.transpose(), extent=(min(n_bins), max(n_bins), min(logl_cutoffs), max(logl_cutoffs)),
                   interpolation='bicubic', aspect='auto', vmin=-vmax, vmax=vmax, cmap=plt.cm.bwr)
        plt.colorbar()
        plt.savefig(outpref+'mean_g2_2d.png')


if __name__ == "__main__":
    import pdb, traceback
    try:
        main(sys.argv)
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
