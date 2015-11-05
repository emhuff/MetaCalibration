#!/usr/bin/env python
import numpy as np
from scipy.special import psi, gamma
from scipy.optimize import root


def eta_est(x, mu_0, sigma_0, nu_0):
    lambda_0 = 1./(sigma_0**2)
    return (nu_0 + 1.)/(nu_0 + lambda_0*((x-mu_0)**2))

def log_eta_est(x, mu_0, sigma_0, nu_0):
    lambda_0 = 1./(sigma_0**2)
    return psi((nu_0 +1)/2.) - np.log( (nu_0 + lambda_0*(x-mu_0)**2 )/2.)

def mu_est(x, mu_0, sigma_0, nu_0):
    eta_0 = eta_est( x,mu_0, sigma_0, nu_0)
    return np.sum(x*eta_0) / np.sum(eta_0)

def sigma_est(x, mu_0, sigma_0, nu_0):
    eta_0 = eta_est(x, mu_0, sigma_0, nu_0)
    sig2 = np.sum((x-mu_0)**2 * eta_0) * 1./len(x)
    return np.sqrt(sig2 )

def nu_root(nu, x, mu_0, sigma_0, nu_0):
    eta_0 = eta_est(x, mu_0, sigma_0, nu_0)
    log_eta_0 = log_eta_est(x, mu_0, sigma_0, nu_0)
    return ( -psi(nu/2.) + np.log(nu/2.) + 1 + np.mean(log_eta_0) - np.mean(eta_0) )

def nu_est_nl(x, mu_0, sigma_0, nu_0):
    nu_sol = root(nu_root, 1, args=(x,mu_0, sigma_0, nu_0))
    return nu_sol.x
    #return 2.0


def estimate_t(x, mu_0 = None, nu_0 = None, sigma_0 = None, n_iter = 10):
    # come up with some intial guesses.
    if mu_0 is None:
        mu_0 = 0.#np.mean(x)
    if nu_0 is None:
        nu_0 = 2
    if sigma_0 is None:
        sigma_0 = 1.#np.std(x)
    # Now iteratively improve on these initial estimates:
    mu_1 = mu_0
    nu_1 = nu_0
    sigma_1 = sigma_0
    for i in xrange(n_iter):
        mu_1 = mu_est(x, mu_1, sigma_1, nu_1)
        sigma_1 = sigma_est(x, mu_1, sigma_1, nu_1)
        nu_1 = nu_est_nl(x, mu_1, sigma_1, nu_1)
    return mu_1, sigma_1, nu_1

def t(x, mu, sigma, nu):
    return gamma((nu+1)/2.) / ( np.sqrt(np.pi * nu) * gamma(nu/2.) ) * (1+ ( (x - mu)/sigma )**2 / nu ) **( - (nu+1)/2.)

def test_problem(n_sample = 100):
    # first, select values of our parameters.
    mu = 1.0
    nu = 2.0
    sigma = 5.0
    # draw some points from the distribution.
    x = mu + sigma * np.random.standard_t(nu,size=n_sample)
    print nu_est_nl(x, mu, sigma, nu)
    print mu_est(x, mu, sigma, nu)
    print sigma_est(x, mu, sigma, nu)
    # Now try to fit these points with our fancy-pants E-M Student's t estimator.
    mu_first, sigma_first, nu_first = estimate_t(x,n_iter=1)
    mu_best, sigma_best, nu_best =  estimate_t(x,n_iter=1000)

    bins = np.linspace(-50,50,100)
    xx = (bins[0:-1] + bins[1:])/2.
    yy_true =  t(xx, mu, sigma, nu)
    yy_est =  t(xx, mu_best, sigma_best, nu_best)
    import matplotlib.pyplot as plt
    plt.plot(xx,yy_true/np.sum(yy_true),label='true')
    plt.plot(xx,yy_est/np.sum(yy_est), label='est')
    
    plt.hist(x,  bins=bins, normed=True, label='data',alpha=0.5)
    plt.legend(loc='best')
    plt.show()
        
def main(argv):
    test_problem()

       
if __name__ == "__main__":
    import pdb, traceback, sys
    try:
        main(sys.argv)
    except:
        thingtype, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)



    
