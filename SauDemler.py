from __future__ import division, print_function
import numpy as np
import matplotlib
import kwant
from matplotlib import pyplot as plt
from matplotlib import pyplot
from scipy import integrate
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.optimize import brentq
import multiprocessing

import sys
import os

defaultcpu=8

try:
    cpus = multiprocessing.cpu_count()
    #np.savetxt('cpusmycomp.txt',[cpus,cpus])
except NotImplementedError:
    cpus = defaultcpu   # arbitrary default

cpus = defaultcpu

relE = 1.e100
absE = 0
bands = 2
itt = 5000

mu = 0.5
alph = 0.6
delt = .5
V = 0.8

c = 3e8
hbar = 6.582119e-13
me = 5.12e8 / c**2
mstar = .15 * me
a0met = 1.5e-8


coeff = hbar / (np.sqrt(2 * mstar) * a0met)



numsites = 2
#index = np.linspace(0, 2 * numsites - 1, 2 * numsites)
index = [0,1,2,3]

### pauli matricies ###
#sigma0 = ta.array([[1., 0.], [0., 1.]])
#sigmax = ta.array([[0., 1.], [1., 0.]])
#sigmay = ta.array([[0., -1j], [1j, 0.]])
#sigmaz = ta.array([[1., 0.], [0., -1.]])
#s0tz = np.kron(sigma0, sigmaz)
#sxt0 = np.kron(sigmax, sigma0)
#s0tx = np.kron(sigma0, sigmax)
#sytz = np.kron(sigmay, sigmaz)
#sxtz = np.kron(sigmax, sigmaz)

sigma0 = np.array([[1., 0.], [0., 1.]])
sigmax = np.array([[0., 1.], [1., 0.]])
sigmay = np.array([[0., -1j], [1j, 0.]])
sigmaz = np.array([[1., 0.], [0., -1.]])

s0 = np.kron(np.identity(numsites), np.identity(2))
sy = np.kron(np.identity(numsites),sigmay)
sz = np.kron(np.identity(numsites),sigmaz)

t0= np.kron(np.identity(2), np.identity(numsites))
tz = np.kron(sigmaz,np.identity(numsites))
tx = np.kron(sigmax,np.identity(numsites))
ty = np.kron(sigmay,np.identity(numsites))


s0t0 = np.dot(s0, t0)
szsy = np.kron(sigmaz, sigmay)


def H0(k):
    return (((coeff * k)**2) - mu) * tz + (alph) * coeff * k * np.dot(sy, tz) + V * sz + delt * tx

def H02(k, V):
    return (((coeff * k)**2) - mu) * tz + (alph) * coeff * k * np.dot(sy, tz) + V * sz + delt * tx

def integrand(H, E):
    return np.linalg.inv(H - E * s0t0)


def singularity(k, e):
    sinVal = delt**4 - 2*delt**2*e**2 + e**4 + 2*alph**2*delt**2*k**2 -2*alph**2*e**2*k**2 + alph**4*k**4 + 2*delt**2*k**4 - 2*e**2*k**4 - 2*alph**2*k**6 + k**8 - 4*delt**2*k**2*mu + 4*e**2*k**2*mu + 4*alph**2*k**4*mu - 4*k**6*mu + 2*delt**2*mu**2 - 2*e**2*mu**2 - 2*alph**2*k**2*mu**2 + 6*k**4*mu**2 - 4*k**2*mu**3 + mu**4 - 2*delt**2*V**2 -2*e**2*V**2 + 2*alph**2*k**2*V**2 - 2*k**4*V**2 + 4*k**2*mu*V**2 -2*mu**2*V**2 + V**4
    return sinVal

####Adjust points later. This is for a specific E #####
def G0(i, j, E):
    #bounds_k=[-3.14,3.14]
    #options={'limit':itt,'epsrel':relE,'epsabs':absE}

    def f(k):
        H = H0(k)
        return integrand(H, E)[i, j]

    #return integrate.quad(f,-np.pi,np.pi,points=[-1.85451,-1.4945,1.4948,1.85451],limit=itt)[0]
    #return integrate.quad(f,-np.pi,np.pi,limit=itt,epsrel=relE,epsabs=absE)
    return coeff * integrate.quad(f, -np.inf/coeff, np.inf/coeff, limit=itt)[0]

def G02(i, j, E, V):
    def f(k):
        H=H02(k, V)
        #return integrand(H,E)[i,j]
        return integrand(H,E)[i, j]
    return coeff * integrate.quad(f, -np.inf/coeff, np.inf/coeff, limit = itt)[0]

def finalvals(En):
    fullG0 = np.dot([[G0(i, j, En) for i in index] for j in index],tz)
    '''
    def solvefunc(g):
        return np.linalg.det(g*s0t0-fullG0)
    return fsolve(solvefunc,-2)
    '''
    sovefunc = np.linalg.eig(fullG0)[0]
    #ens = np.repeat(En,numsites)
    return sovefunc

def finalvals3(En, V):
    fullG0 = np.dot([[G02(i, j, En, V) for i in index] for j in index],tz)
    sovefunc = np.linalg.eig(fullG0)[0]
    return sovefunc

#print(finalvals3(.05, .8))

'''
def finalvals(k,V,g):
    H=H0g(k,V,g)
    fullG0 = np.dot([[G0(i,j,En) for i in index] for j in index],tz)
    #sovefunc= np.linalg.eig(fullG0)[0]
    sovefunc=np.linalg.eig(fullG0)[0]
    #ens = np.repeat(En,numsites)
    return sovefunc
'''

Energies = np.linspace(-.09282, .09282, 100)
pots = np.linspace(-1, 2, 100)

#print(brentq(finalvals,-.2,.2,args=.2))

#################################################################################################
# Creates Diretories for files
#################################################################################################

cwd = os.getcwd()

dirName = os.path.join(cwd, 'data', 'mu{mu}_alph{alph}_delt{delt}'.format(mu=mu,alph=alph,delt=delt, fmt='%s'))

if not os.path.exists(dirName):
    os.makedirs(dirName)

#################################################################################################


def plot_spectrum():
    pool = multiprocessing.Pool(processes=cpus)
    spectrum = pool.map(finalvals, Energies)
    fname = os.path.join(dirName,"pot{pot}.txt".format(pot = V, fmt = '%s'))
    enFname = os.path.join(dirName,"energyValues_pot{pot}.txt".format(pot = V, fmt = '%s'))
    np.savetxt(fname, spectrum)
    np.savetxt(enFname, Energies)

plot_spectrum()

def plot_spectrum2():
    spectrum = [[finalvals3(En, V) for En in Energies] for V in pots]
    #spectrum = pool.map(finalvals3,pots)
    fname = "/home/gorot00/Desktop/SauDemlerVTest.txt"
    np.savetxt(fname.format(pot = V), spectrum, fmt = '%s')
#plot_spectrum2()

#########################################################################################
###########                   Figure 2 plots                      #######################
#########################################################################################


#def nanowire_chain(L=None, t=1, mu=0.1, delta=0.1, B=0.1, alpha=0,g=-.75):
def nanowire_chain(t, mu, delta, B, alpha,g,L=None):
    """
    Return finalized kwant system for a nanowire chain.
    For L=None function returns infinite (TranslationalSymmetry)
    system. If L is set to positive integer the system is finite.
    """
    lat = kwant.lattice.chain()

    if L is None:
        sys = kwant.Builder(kwant.TranslationalSymmetry((-1,)))
        L = 1
    else:
        sys = kwant.Builder()

    # onsite terms
    for x in xrange(L):
        sys[lat(x)] = (2*t - mu) * tz + B * sz + delta * tx + g*tz

    # hopping terms
    sys[kwant.HoppingKind((1,), lat)] = -t * tz - 0.5 * 1j * alpha * szsy

    return sys.finalized()

def find_gap(sys, resolution=1e-4):
    """Find gap in a system by doing a binary search in energy."""
    # This tells us if there are modes at a certain energy.
    if len(sys.modes(energy=0)[0].momenta):
        return 0
    gap = step = min(abs(kwant.physics.Bands(sys)(k=0))) / 2
    while step > resolution:
        step /= 2
        if len(sys.modes(gap)[0].momenta):
            gap -= step
        else:
            gap += step

    return gap

def spinorbit_band_gap(mu, t, delta,alpha,gs, B_vals):
    B_crit = np.sqrt((mu)**2 + delta**2)
    gaps=[]

    # very slow, but understandable
    for g in gs:
        temp_gaps = []
        for B in B_vals:
            sys = nanowire_chain(t = t, mu = mu, delta = delta, alpha = alpha, B = (B - B_crit), g = 0)
            en=find_gap(sys)
            print(B)
            sys2 = nanowire_chain(t = t, mu = mu, delta = delta, alpha = alpha, B = (B - B_crit), g = g * (1 / finalvals3(en, (B - B_crit))[1].real))

            temp_gaps.append(find_gap(sys2))
        gaps.append(np.array(temp_gaps))
    fig = plt.figure()


    for gap, g in zip(gaps, gs):
        plt.plot(B_vals, gap, label= (r'$g = %1.1f $' % g))
    plt.xlabel('$B$')
    plt.ylabel('Band gap')
    plt.xticks(np.linspace(B_vals[0], B_vals[-1], 4))
    ylim = [0.0, 0.14]
    #ylim = [0.0, 1]
    plt.ylim(ylim)
    plt.yticks([0, 0.05, 0.1])
    plt.legend(bbox_to_anchor=(0.0, 0.75, 1.0, .09), loc=3,
               ncol=2, mode="expand", borderaxespad=0., prop={'size':16})
    #plt.subplots_adjust(top = 0.2)
    plt.title(r'$\mu = %1.3f$, $\Delta = %1.1f$' % (mu, delta))
    # Add a vertical line to show where the topological region begins:
    #plt.plot([0,0], ylim, 'r--')
    plt.plot([B_crit, B_crit], ylim, 'r--')
    return plt.show(fig)

#mu,alph,V,delt=.5,.6,.8,.5

#(mu, t, delta,alpha, B_vals)
B_vals = np.linspace(0, 1.3, 101)
gs = [1]

'''
spinorbit_band_gap(0.050, 1, 0.1,0.3,gs, B_vals)
#spinorbit_band_gap(.5, 1, 0.5,.6,gs, B_vals)

'''









