# Simple routine to make images of PSFs.  Stolen directly from GREAT3 code base.
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import sys

# Absolute path to GREAT3 analysis code
analysis_path = '/home/rmandelb/git/great3-private/analysis'
# How to get back to MetaCal repo (abs path)
paper_path = '/home/rmandelb/git/MetaCalibration' 
fig_dir = 'Plots' # directory for figures.
# Scratch disk where these sims live
sim_parent = '/nfs/nas-0-9/rmandelb'

if __name__ == "__main__":

    # A bunch of things depend on us having access to the routines in ../../analysis/.
    # So let's go there.
    os.chdir(analysis_path)
    sys.path.append('.')
    #import loader
    #import submissions
    import diagnostics

    # plot PSFs themselves:  Let's choose some subfields, and plot for RGC-noaber, RGC-aber (large aberrations in all fields)
    subfields = [0, 37, 142]
    for subfield in subfields:
        plot = diagnostics.show_psf(subfield, os.path.join(sim_parent, 'great3-eric-noaber'), 'real_galaxy',
                                    'ground', 'constant', vmin=-0.003, vmax=0.042,
                                    show_colorbar=False)
        plot.save(os.path.join(paper_path, fig_dir, 'rgc_noaber_psf_%03d.pdf'%subfield))

        plot = diagnostics.show_psf(subfield, os.path.join(sim_parent, 'great3-eric-fixedaber'), 'real_galaxy',
                                    'ground', 'constant', vmin=-0.003, vmax=0.042,
                                    show_colorbar=False)
        plot.save(os.path.join(paper_path, fig_dir, 'rgc_fixedaber_psf_%03d.pdf'%subfield))
