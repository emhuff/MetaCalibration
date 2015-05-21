import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
from scipy.optimize import curve_fit

def shear_model(x, m, a, c):
    # x should be 3 x N, where 0=gtrue, 1=epsf, 2=const
    return m*x[0,:] + a*x[1,:] + c*x[2,:]
    

truthfile = 'cgc-truthtable.txt'

n_bins = [25, 50, 60, 70, 75, 85, 100, 125, 150]
percentile_vals = [0.5, 1., 2., 3., 4., 5., 6., 8., 10., 12., 14., 16., 20.]
filepref = 'outputs/output-cgc-nosymm-'
filesuff = '.dat'
rootdir = './' # to be passed to shear_ensemble_est.py
mc_type = 'regauss' # to be passed to shear_ensemble_est.py
outpref = 'nbins-'

mean_g1 = np.zeros((len(n_bins), len(percentile_vals)))
mean_g2 = np.zeros((len(n_bins), len(percentile_vals)))
sig_g1 = np.zeros((len(n_bins), len(percentile_vals)))
sig_g2 = np.zeros((len(n_bins), len(percentile_vals)))
m1 = np.zeros((len(n_bins), len(percentile_vals)))
m2 = np.zeros((len(n_bins), len(percentile_vals)))
sig_m1 = np.zeros((len(n_bins), len(percentile_vals)))
sig_m2 = np.zeros((len(n_bins), len(percentile_vals)))
a1 = np.zeros((len(n_bins), len(percentile_vals)))
a2 = np.zeros((len(n_bins), len(percentile_vals)))
sig_a1 = np.zeros((len(n_bins), len(percentile_vals)))
sig_a2 = np.zeros((len(n_bins), len(percentile_vals)))
c1 = np.zeros((len(n_bins), len(percentile_vals)))
c2 = np.zeros((len(n_bins), len(percentile_vals)))
sig_c1 = np.zeros((len(n_bins), len(percentile_vals)))
sig_c2 = np.zeros((len(n_bins), len(percentile_vals)))

for n_indx, n in enumerate(n_bins):
    # construct filename
    filename = '%s%d%s'%(filepref,n,filesuff)

    # check for existence; if not, create
    if not os.path.exists(filename):
        print 'Generating file %s'%filename
        tmp_command = 'python shear_ensemble_est.py --path %s --mc_type %s -o %s -n %d'%(rootdir,mc_type,filename,n)
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
        logl_cutoffs = []
        for perc in percentile_vals:
            logl_cutoffs.append(np.percentile(logl, perc))

    # read in truth table
    dat = np.loadtxt(truthfile).transpose()
    field_id_truth = dat[0,:]
    g1_true = dat[1,:]
    g2_true = dat[2,:]

    if not np.array_equal(field_id, field_id_truth):
        raise RuntimeError('Subfield indices do not match!')

    for logl_indx, logl_cut in enumerate(logl_cutoffs):
        print "Log likelihood cutoff: ",logl_cut
        to_save_1 = logl1 > logl_cut
        use_g1_opt = g1_opt[to_save_1]
        use_g1_var = g1_var[to_save_1]
        use_psf_e1 = psf_e1[to_save_1]
        use_g1_true = g1_true[to_save_1]

        to_save_2 = logl2 > logl_cut
        use_g2_opt = g2_opt[to_save_2]
        use_g2_var = g2_var[to_save_2]
        use_psf_e2 = psf_e2[to_save_2]
        use_g2_true = g2_true[to_save_2]

        print "Using ",len(use_g1_true),' and ',len(use_g2_true),' for g1 and g2'

        # compute and store <gamma>
        mean_g1[n_indx][logl_indx]=np.mean(use_g1_opt)
        mean_g2[n_indx][logl_indx]=np.mean(use_g2_opt)

        # store sigma_gamma
        sig_g1[n_indx][logl_indx]=np.mean(np.sqrt(use_g1_var))
        sig_g2[n_indx][logl_indx]=np.mean(np.sqrt(use_g2_var))

        # compute (roughly) and store m, a
        # Our equation is g_obs - g_true = m*g_true + a*e_psf
        # Our "A" matrix that lstsq wants should be like this:
        A = np.column_stack([use_g1_true, use_psf_e1, np.ones_like(use_psf_e1)]).transpose()
        # And the B vector is just the LHS
        B = use_g1_opt - use_g1_true
        ret_val, covar_1 = curve_fit(shear_model, A, B, sigma=np.sqrt(use_g1_var))
        m1[n_indx][logl_indx]=ret_val[0]
        a1[n_indx][logl_indx]=ret_val[1]
        c1[n_indx][logl_indx]=ret_val[2]
        sig_m1[n_indx][logl_indx]=np.sqrt(covar_1[0][0])
        sig_a1[n_indx][logl_indx]=np.sqrt(covar_1[1][1])
        sig_c1[n_indx][logl_indx]=np.sqrt(covar_1[2][2])
        A = np.column_stack([use_g2_true, use_psf_e2, np.ones_like(use_psf_e2)]).transpose()
        B = use_g2_opt - use_g2_true
        ret_val, covar_2 = curve_fit(shear_model, A, B, sigma=np.sqrt(use_g2_var))
        m2[n_indx][logl_indx]=ret_val[0]
        a2[n_indx][logl_indx]=ret_val[1]
        c2[n_indx][logl_indx]=ret_val[2]
        sig_m2[n_indx][logl_indx]=np.sqrt(covar_2[0][0])
        sig_a2[n_indx][logl_indx]=np.sqrt(covar_2[1][1])
        sig_c2[n_indx][logl_indx]=np.sqrt(covar_2[2][2])

if 0:
    # Plot m vs. n_bins
    fig = plt.figure()
    ax = fig.add_subplot(321)
    ax.errorbar(n_bins, 100*np.array(m1), yerr=100*np.array(sig_m1), label=r'$m_1$')
    ax.errorbar(n_bins, 100*np.array(m2), yerr=100*np.array(sig_m2), label=r'$m_2$')
    ax.plot(n_bins, np.zeros_like(n_bins), 'k--')
    plt.xlim((0.9*min(n_bins), 1.1*max(n_bins)))
#ax.set_xlabel(r'$N_{\rm bins}$')
    ax.set_ylabel(r'$10^2 m$')

# Plot a vs. n_bins
    ax = fig.add_subplot(322)
    ax.errorbar(n_bins, 1000*np.array(a1), yerr=1000*np.array(sig_a1), label=r'$a_1$')
    ax.errorbar(n_bins, 1000*np.array(a2), yerr=1000*np.array(sig_a2), label=r'$a_2$')
    ax.plot(n_bins, np.zeros_like(n_bins), 'k--')
#ax.set_xlabel(r'$N_{\rm bins}$')
    plt.xlim((0.9*min(n_bins), 1.1*max(n_bins)))
    ax.set_ylabel(r'$10^{3} a$')

# Plot c vs. n_bins
    ax = fig.add_subplot(325)
    ax.errorbar(n_bins, 1000*np.array(c1), yerr=1000*np.array(sig_c1), label=r'$c_1$')
    ax.errorbar(n_bins, 1000*np.array(c2), yerr=1000*np.array(sig_c2), label=r'$c_2$')
    ax.plot(n_bins, np.zeros_like(n_bins), 'k--')
    ax.set_xlabel(r'$N_{\rm bins}$')
    plt.xlim((0.9*min(n_bins), 1.1*max(n_bins)))
    ax.set_ylabel(r'$10^{3} c$')

# Plot <g> vs. n_bins
    ax = fig.add_subplot(323)
    ax.plot(n_bins, 1000*np.array(mean_g1), 'rx', label=r'$\langle\gamma_1\rangle$')
    ax.plot(n_bins, 1000*np.array(mean_g2), 'bo', label=r'$\langle\gamma_2\rangle$')
    ax.plot(n_bins, 1000*np.mean(g1_true)*np.ones_like(n_bins), 'r--')
    ax.plot(n_bins, 1000*np.mean(g2_true)*np.ones_like(n_bins), 'b--')
#ax.set_xlabel(r'$N_{\rm bins}$')
    plt.xlim((0.9*min(n_bins), 1.1*max(n_bins)))
    ax.set_ylabel(r'$10^3 \langle\gamma\rangle$')

# Plot sigma_g vs. n_bins
    ax = fig.add_subplot(324)
    ax.plot(n_bins, 1000*np.array(sig_g1), 'rx', label=r'$\langle\sigma_{\gamma,1}\rangle$')
    ax.plot(n_bins, 1000*np.array(sig_g2), 'bo', label=r'$\langle\sigma_{\gamma,2}\rangle$')
    ax.set_xlabel(r'$N_{\rm bins}$')
    plt.xlim((0.9*min(n_bins), 1.1*max(n_bins)))
    ax.set_ylabel(r'$10^3 \sigma_\gamma$')

    plt.tight_layout()
    plt.savefig(outpref+mc_type+'.png')

else:
    # make 2d plot of <shear residual> vs. n, logL
    fig = plt.figure()
    vmax = max(np.max(c1), -np.min(c1))
    plt.imshow(c1.transpose(), extent=(min(n_bins), max(n_bins), min(logl_cutoffs), max(logl_cutoffs)),
               interpolation='bicubic', aspect='auto', vmin=-vmax, vmax=vmax, cmap=plt.cm.bwr)
    plt.colorbar()
    plt.savefig('mean_g1_2d.png')
    fig = plt.figure()
    plt.imshow(c2.transpose(), extent=(min(n_bins), max(n_bins), min(logl_cutoffs), max(logl_cutoffs)),
               interpolation='bicubic', aspect='auto', vmin=-vmax, vmax=vmax, cmap=plt.cm.bwr)
    plt.colorbar()
    plt.savefig('mean_g2_2d.png')

    # make 2d plot of m vs. n, logL
    fig = plt.figure()
    vmax = max(np.max(m1), -np.min(m1))
    plt.imshow(m1.transpose(), extent=(min(n_bins), max(n_bins), min(logl_cutoffs), max(logl_cutoffs)),
               interpolation='bicubic', aspect='auto', vmin=-vmax, vmax=vmax, cmap=plt.cm.bwr)
    plt.colorbar()
    plt.savefig('mean_m1_2d.png')
    fig = plt.figure()
    plt.imshow(m2.transpose(), extent=(min(n_bins), max(n_bins), min(logl_cutoffs), max(logl_cutoffs)),
               interpolation='bicubic', aspect='auto', vmin=-vmax, vmax=vmax, cmap=plt.cm.bwr)
    plt.colorbar()
    plt.savefig('mean_m2_2d.png')

    # make 2d plot of a vs. n, logL
    fig = plt.figure()
    vmax = max(np.max(m1), -np.min(m1))
    plt.imshow(a1.transpose(), extent=(min(n_bins), max(n_bins), min(logl_cutoffs), max(logl_cutoffs)),
               interpolation='bicubic', aspect='auto', vmin=-vmax, vmax=vmax, cmap=plt.cm.bwr)
    plt.colorbar()
    plt.savefig('mean_a1_2d.png')
    fig = plt.figure()
    plt.imshow(a2.transpose(), extent=(min(n_bins), max(n_bins), min(logl_cutoffs), max(logl_cutoffs)),
               interpolation='bicubic', aspect='auto', vmin=-vmax, vmax=vmax, cmap=plt.cm.bwr)
    plt.colorbar()
    plt.savefig('mean_a2_2d.png')
