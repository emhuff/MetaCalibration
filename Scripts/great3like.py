"""
This script makes a GREAT3-like plot for the paper.

If the format of final_field_fit_coefficients.txt changes, then a lot of the file-parsing code has
to change too.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True

# define constants, files, etc.
infile = 'final_field_fit_coefficients.txt'
outfile1 = 'great3_before_after'
outfile2 = 'great3_before'
outfile3 = 'great3_after'
outsuff = '.png'
a_lim = (-0.1,0.1)
m_lim = (-1.,1.)
# components: average them or ...?
components = ['1','2','both']
# make dicts with translation of codes to branches and methods
info_dict = {
    'regauss':{'branch':'CGC','method':'Regauss','marker':'s','color':(0.1,0.1,0.1),'ls':'-'},
    'ksb':{'branch':'CGC','method':'KSB','marker':'s','color':(0.9,0.1,0.1),'ls':'-'},
    'moments':{'branch':'CGC','method':'Moments','marker':'s','color':(0.9,0.1,0.9),'ls':'-'},
    'noaber-regauss':{'branch':'CGC-noaber','method':'Regauss','marker':'o','color':(0.1,0.1,0.1),'ls':'-'},
    'rgc-regauss':{'branch':'RGC','method':'Regauss','marker':'v','color':(0.1,0.1,0.1),'ls':'-'},
    'rgc-noaber-regauss':{'branch':'RGC-noaber','method':'Regauss','marker':'^','color':(0.1,0.1,0.1),'ls':'-'},
    'rgc-fixedaber-regauss':{'branch':'RGC-fixedaber','method':'Regauss','marker':'D','color':(0.1,0.1,0.1),'ls':'-'},
    'rgc-ksb':{'branch':'RGC','method':'KSB','marker':'v','color':(0.9,0.1,0.1),'ls':'-'},
}
m_targ = 0.002

# read in data
dat = np.genfromtxt(infile, dtype=None)
# now we have an unstructured array
# yank out the methods
n = len(dat)
method_list = []
new_array = np.zeros((n, len(dat[0])-1)).astype(float)
for ind in range(n):
    method_list.append(dat[ind][0])
    for ind2 in range(1,len(dat[ind])-1):
        new_array[ind,ind2-1] = dat[ind][ind2]

# set up plot: try showing before/after on same plot

# now try separate before and after plots

# before plot
for component in components:
    fig = plt.figure()

    if component == 'both':
        a_vals = 0.5*(new_array[:,1]+new_array[:,7])
        a_err_vals = 0.5*np.sqrt(new_array[:,4]**2 + new_array[:,10]**2)
        m_vals = 0.5*(new_array[:,0]+new_array[:,6])
        m_err_vals = 0.5*np.sqrt(new_array[:,3]**2 + new_array[:,9]**2)
    elif component == '1':
        a_vals = new_array[:,1]
        a_err_vals =new_array[:,4]
        m_vals = new_array[:,0]
        m_err_vals = new_array[:,3]
    else:
        a_vals = new_array[:,7]
        a_err_vals = new_array[:,10]
        m_vals = new_array[:,6]
        m_err_vals = new_array[:,9]
    min_a = min(a_vals-a_err_vals)
    max_a = max(a_vals+a_err_vals)
    min_m = min(m_vals-m_err_vals)
    max_m = max(m_vals+m_err_vals)
    for ind in range(n):
        mstr = method_list[ind]
        if 'moments' not in mstr:
            plt.errorbar([a_vals[ind]], [m_vals[ind]], xerr=[a_err_vals[ind]], yerr=[m_err_vals[ind]],
                         marker=info_dict[mstr]['marker'], color=info_dict[mstr]['color'],
                         label=info_dict[mstr]['method']+' ('+info_dict[mstr]['branch']+')')
            #if 'ksb' in info_dict[mstr]['method'].lower():
            #    print info_dict[mstr]['method'], m_vals[ind], a_vals[ind]
    #a_lim = (min_a, max_a)
    #m_lim = (min_m, max_m)
    zero_arr = np.zeros(2)
    plt.plot(zero_arr, m_lim, linestyle='--', color='k')
    plt.plot(a_lim, zero_arr, linestyle='--', color='k')
    plt.fill_between(a_lim, (-m_targ, -m_targ), (m_targ, m_targ), facecolor='lightgrey', edgecolor='None')

    plt.xlim(a_lim)
    plt.ylim(m_lim)
    if component == 'both':
        plt.xlabel(r'$\langle a\rangle$')
        plt.ylabel(r'$\langle m\rangle$')
    elif component == '1':
        plt.xlabel(r'$a_1$')
        plt.ylabel(r'$m_1$')
    elif component == '2':
        plt.xlabel(r'$a_2$')
        plt.ylabel(r'$m_2$')

    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, numpoints=1, scatterpoints=1, prop={'size':10})
    plt.subplots_adjust(left=0.12, right=0.7)

    plt.title('Before Metacalibration')
    plt.xscale('symlog', linthreshx=m_targ)
    plt.yscale('symlog', linthreshy=m_targ)
    plt.plot((m_targ, m_targ), m_lim, linestyle='-.', color='k')
    plt.plot((-m_targ, -m_targ), m_lim, linestyle='-.', color='k')
    plt.plot(a_lim, (m_targ, m_targ), linestyle='-.', color='k')
    plt.plot(a_lim, (-m_targ, -m_targ), linestyle='-.', color='k')

    outname = outfile2 + '_' + component + outsuff
    print 'saving to file ',outname
    plt.savefig(outname)

# after plot should include moments, too
for component in components:
    fig = plt.figure()

    if component == 'both':
        a_vals = 0.5*(new_array[:,13]+new_array[:,19])
        a_err_vals = 0.5*np.sqrt(new_array[:,16]**2 + new_array[:,22]**2)
        m_vals = 0.5*(new_array[:,12]+new_array[:,18])
        m_err_vals = 0.5*np.sqrt(new_array[:,15]**2 + new_array[:,21]**2)
    elif component == '1':
        a_vals = new_array[:,13]
        a_err_vals =new_array[:,16]
        m_vals = new_array[:,12]
        m_err_vals = new_array[:,15]
    else:
        a_vals = new_array[:,19]
        a_err_vals = new_array[:,22]
        m_vals = new_array[:,18]
        m_err_vals = new_array[:,21]
    min_a = min(a_vals-a_err_vals)
    max_a = max(a_vals+a_err_vals)
    min_m = min(m_vals-m_err_vals)
    max_m = max(m_vals+m_err_vals)
    for ind in range(n):
        mstr = method_list[ind]
        plt.errorbar([a_vals[ind]], [m_vals[ind]], xerr=[a_err_vals[ind]], yerr=[m_err_vals[ind]],
                     marker=info_dict[mstr]['marker'], color=info_dict[mstr]['color'],
                     label=info_dict[mstr]['method']+' ('+info_dict[mstr]['branch']+')')
    #a_lim = (min_a, max_a)
    #m_lim = (min_m, max_m)
    zero_arr = np.zeros(2)
    plt.plot(zero_arr, m_lim, linestyle='--', color='k')
    plt.plot(a_lim, zero_arr, linestyle='--', color='k')
    plt.fill_between(a_lim, (-m_targ, -m_targ), (m_targ, m_targ), facecolor='lightgrey', edgecolor='None')

    plt.xlim(a_lim)
    plt.ylim(m_lim)
    if component == 'both':
        plt.xlabel(r'$\langle a\rangle$')
        plt.ylabel(r'$\langle m\rangle$')
    elif component == '1':
        plt.xlabel(r'$a_1$')
        plt.ylabel(r'$m_1$')
    elif component == '2':
        plt.xlabel(r'$a_2$')
        plt.ylabel(r'$m_2$')

    plt.legend(bbox_to_anchor=(1.01, 1), loc=2, numpoints=1, scatterpoints=1, prop={'size':10})
    plt.subplots_adjust(left=0.12, right=0.7)

    plt.title('After Metacalibration')
    plt.xscale('symlog', linthreshx=m_targ)
    plt.yscale('symlog', linthreshy=m_targ)
    plt.plot((m_targ, m_targ), m_lim, linestyle='-.', color='k')
    plt.plot((-m_targ, -m_targ), m_lim, linestyle='-.', color='k')
    plt.plot(a_lim, (m_targ, m_targ), linestyle='-.', color='k')
    plt.plot(a_lim, (-m_targ, -m_targ), linestyle='-.', color='k')

    outname = outfile3 + '_' + component + outsuff
    print 'saving to file ',outname
    plt.savefig(outname)
