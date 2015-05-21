import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import os
import subprocess

truthfile = 'cgc-noaber-truthtable.txt'

n_bins = [25, 50, 75, 85, 100, 150, 200]
filepref = 'outputs/output-cgc-noaber-nosymm-'
filesuff = '.dat'
rootdir = './' # to be passed to shear_ensemble_est.py
mc_type = 'noaber-regauss' # to be passed to shear_ensemble_est.py

mean_g1 = []
mean_g2 = []
sig_g1 = []
sig_g2 = []
m1 = []
m2 = []
a1 = []
a2 = []
for n in n_bins:
    # construct filename
    filename = '%s%d%s'%(filepref,n,filesuff)

    # check for existence; if not, create
    if not os.path.exists(filename):
        print 'Generating file %s'%filename
        tmp_command = 'python shear_ensemble_est.py %s %s %s %d'%(rootdir,mc_type,filename,n)
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

    # read in truth table
    dat = np.loadtxt(truthfile).transpose()
    field_id_truth = dat[0,:]
    g1_true = dat[1,:]
    g2_true = dat[2,:]

    if not np.array_equal(field_id, field_id_truth):
        raise RuntimeError('Subfield indices do not match!')

    # compute and store <gamma>
    mean_g1.append(np.mean(g1_opt))
    mean_g2.append(np.mean(g2_opt))

    # store sigma_gamma
    sig_g1.append(np.mean(np.sqrt(g1_var)))
    sig_g2.append(np.mean(np.sqrt(g2_var)))

    # compute (roughly) and store m, a
    # Our equation is g_obs - g_true = m*g_true + a*e_psf
    # Our "A" matrix that lstsq wants should be like this:
    A = np.column_stack([g1_true, psf_e1, np.ones_like(psf_e1)])
    # And the B vector is just the LHS
    B = g1_opt - g1_true
    ret_val = scipy.linalg.lstsq(A, B)
    m1.append(ret_val[0][0])
    a1.append(ret_val[0][1])
    A = np.column_stack([g2_true, psf_e2, np.ones_like(psf_e2)])
    B = g2_opt - g2_true
    ret_val = scipy.linalg.lstsq(A, B)
    m2.append(ret_val[0][0])
    a2.append(ret_val[0][1])

# Plot m vs. n_bins
fig = plt.figure()
ax = fig.add_subplot(221)
ax.plot(n_bins, 100*np.array(m1), 'rx', label=r'$m_1$')
ax.plot(n_bins, 100*np.array(m2), 'bo', label=r'$m_2$')
ax.plot(n_bins, np.zeros_like(n_bins), 'k--')
#ax.set_xlabel(r'$N_{\rm bins}$')
ax.set_ylabel(r'$10^2 m$')

# Plot a vs. n_bins
ax = fig.add_subplot(222)
ax.plot(n_bins, 1000*np.array(a1), 'rx', label=r'$a_1$')
ax.plot(n_bins, 1000*np.array(a2), 'bo', label=r'$a_2$')
ax.plot(n_bins, np.zeros_like(n_bins), 'k--')
#ax.set_xlabel(r'$N_{\rm bins}$')
ax.set_ylabel(r'$10^{3} a$')

# Plot <g> vs. n_bins
ax = fig.add_subplot(223)
ax.plot(n_bins, 1000*np.array(mean_g1), 'rx', label=r'$\langle\gamma_1\rangle$')
ax.plot(n_bins, 1000*np.array(mean_g2), 'bo', label=r'$\langle\gamma_2\rangle$')
ax.plot(n_bins, 1000*np.mean(g1_true)*np.ones_like(n_bins), 'r--')
ax.plot(n_bins, 1000*np.mean(g2_true)*np.ones_like(n_bins), 'b--')
ax.set_xlabel(r'$N_{\rm bins}$')
ax.set_ylabel(r'$10^3 \langle\gamma\rangle$')

# Plot sigma_g vs. n_bins
ax = fig.add_subplot(224)
ax.plot(n_bins, 1000*np.array(sig_g1), 'rx', label=r'$\langle\sigma_{\gamma,1}\rangle$')
ax.plot(n_bins, 1000*np.array(sig_g2), 'bo', label=r'$\langle\sigma_{\gamma,2}\rangle$')
ax.set_xlabel(r'$N_{\rm bins}$')
ax.set_ylabel(r'$10^3 \sigma_\gamma$')

plt.tight_layout()
plt.savefig('nbins.png')
