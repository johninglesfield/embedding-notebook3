# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 11:24:43 2016

@author: johninglesfield
"""
import cmath
import numpy as np
from numpy import sqrt, sin, cos, pi
from scipy.integrate import quad, dblquad
from scipy.linalg import inv
def corner_matrices(param):
    """
    created Monday 11 July 2016
    Electron confined by circular corner: evaluates Hamiltonian and 
    overlap integrals.
    Input parameters: V, confining potential; r1, inner radius; r2, outer
    radius; D, defines basis functions; N, number of basis functions in
    each direction.
    Function returns Hamiltonian and overlap matrices, and al, be, which
    define basis functions.
    """
    global V,r1,r2,D,N
    V,r1,r2,D,N=param
    sigma=sqrt(0.5*V).real
    ab_array=np.zeros((N,N),dtype=int)
    for i in range(N):
        ab_array[i,]=range(N)
    al=ab_array.flatten(order='C')
    be=ab_array.flatten(order='F')
    ham=np.zeros((N*N,N*N))
    ovlp=np.zeros((N*N,N*N))
    for i in range(N*N):
        m=[al[i],be[i]]
        for j in range(N*N):
            n=[al[j],be[j]]
            ovlp[i,j]=dblquad(corner_ovlp_int,r1,r2,lambda x:0.0,\
            lambda x:0.5*pi,args=(m,n))[0]
            kinen=dblquad(corner_kinen_int,r1,r2,lambda x:0.0,\
            lambda x:0.5*pi,args=(m,n))[0]
            confine=sigma*(quad(corner_ovlp_int,0.0,0.5*pi,\
            args=(r1,m,n))[0]+quad(corner_ovlp_int,0.0,0.5*pi,\
            args=(r2,m,n))[0])
            ham[i,j]=kinen+confine
        if (i+1)%5==0:
            print '%s %3d %s %3d' % ('Number of integrals done =',i+1,\
            'x',N*N)
    print '%s %3d %s %3d' % ('Number of integrals done =',i+1,'x',N*N)
    return ham,ovlp,al,be

def corner_ovlp_int(theta,r,m,n):
    """
    created Monday 11 July 2016
    Electron confined by circular corner: evaluates integrand for overlap
    integral for basis functions defined by m and n. Arguments theta, 
    r are circular polar coordinates. 
    """
    fac=pi/D
    x=r*cos(theta)+0.5*(D-r2)
    y=r*sin(theta)+0.5*(D-r2)
    chi_i=cos(fac*m[0]*x)*cos(fac*m[1]*y)
    chi_j=cos(fac*n[0]*x)*cos(fac*n[1]*y)
    return r*chi_i*chi_j
    
def corner_kinen_int(theta,r,m,n):
    """
    created Monday 11 July 2016
    Electron confined by circular corner: evaluates integrand for kinetic
    energy matrix element, between basis functions defined by m and n
    Arguments theta, r are circular polar coordinates. 
    """
    fac=pi/D
    x=r*cos(theta)+0.5*(D-r2)
    y=r*sin(theta)+0.5*(D-r2)
    grad_chi_i=np.array([-m[0]*sin(fac*m[0]*x)*cos(fac*m[1]*y),\
    -m[1]*cos(fac*m[0]*x)*sin(fac*m[1]*y)])
    grad_chi_j=np.array([-n[0]*sin(fac*n[0]*x)*cos(fac*n[1]*y),\
    -n[1]*cos(fac*n[0]*x)*sin(fac*n[1]*y)])
    return 0.5*r*fac*fac*np.dot(grad_chi_i,grad_chi_j)
    
def corner_embed_matrices(n_emb,al,be):
    """
    created Monday 11 July 2016
    Electron confined by circular corner: evaluates integral of waveguide
    function x part of basis function.
    """
    x=0.5*(D-r2)
    cs=[cos(pi*i*x/D) for i in range(N)]
    emb_int=np.zeros((n_emb,N))
    in_ovlp=np.zeros((n_emb,N*N))
    out_ovlp=np.zeros((n_emb,N*N))
    for p in range(n_emb):
        for m in range(N):    
            emb_int[p,m]=quad(corner_embed_int,r1,r2,args=(p+1,m))[0]
        for i in range(N*N):
            in_ovlp[p,i]=emb_int[p,be[i]]*cs[al[i]]
            out_ovlp[p,i]=emb_int[p,al[i]]*cs[be[i]]
    return in_ovlp,out_ovlp
            
def corner_embed_int(r,p,m):
    """
    created Monday 11 July 2016
    Electron confined by circular corner: waveguide function x part of
    basis function.
    """
    w=r2-r1
    x=r+0.5*(D-r2)
    xp=r-r1
    return sin(p*pi*xp/w)*cos(m*pi*x/D)    
       
def corner_green(energy,ham,ovlp,n_emb,in_ovlp,out_ovlp): 
    """
    created Tuesday 11 July 2016
    Electron confined by circular corner: constructs Green function of
    circular corner embedded onto straight waveguides. Returns Green 
    function, evaluated at real energy. 
    """
    w=r2-r1
    energy=complex(energy,0.0)
    embed=np.zeros((N*N,N*N),complex)
    for p in range(n_emb):
        sigma=complex(0.0,-1.0)*cmath.sqrt(2.0*energy-((p+1)*pi/w)**2)/w
        embed=embed+sigma*(np.outer(in_ovlp[p,],in_ovlp[p,])
                          +np.outer(out_ovlp[p,],out_ovlp[p,]))
    green=ham+embed-energy*ovlp
    green=-inv(green)
    return green
    
def corner_current(psi,xp,yp,al,be):
    """
    created Thursday 4 August 2016
    Electron confined by circular corner: constructs real-space
    wave-function and current at point xp, yp for a state in the 
    continuum, with matrix elements given by input psi.
    """
    if xp*xp+yp*yp>r1*r1 and xp*xp+yp*yp<r2*r2:
        x=xp+0.5*(D-r2)
        y=yp+0.5*(D-r2)
        phi=0.0
        dphi_x=0.0
        dphi_y=0.0
        for i in range(N*N):
            phi=phi+psi[i]*cos(pi*al[i]*x/D)*cos(pi*be[i]*y/D)
            dphi_x=dphi_x-psi[i]*pi*al[i]*sin(pi*al[i]*x/D)*\
            cos(pi*be[i]*y/D)/D
            dphi_y=dphi_y-psi[i]*pi*be[i]*cos(pi*al[i]*x/D)*\
            sin(pi*be[i]*y/D)/D
        Psi=phi
        current_x=(np.conj(phi)*dphi_x).imag
        current_y=(np.conj(phi)*dphi_y).imag
    else:
        Psi=complex(0.0,0.0)
        current_x=0.0
        current_y=0.0
    return Psi,current_x,current_y
   
def corner_transmission(E,ham,ovlp,n_emb,in_ovlp,out_ovlp,input_channel,
                        output_channel):
    """
    created Thursday 4 August 2016
    Electron confined by circular corner: calculates transmission probability
    for a given energy, input and output channel.
    """
    E_input=0.5*((input_channel+1)*pi/(r2-r1))**2
    E_output=0.5*((output_channel+1)*pi/(r2-r1))**2
    if E<E_input or E<E_output:
        T=0.0
    else:
        kz_input=sqrt(2.0*(E-E_input))
        kz_output=sqrt(2.0*(E-E_output))
        input_current=0.5*(r2-r1)*kz_input
        sigma_input=-1j*kz_input/(r2-r1)
        sigma_output=-1j*kz_output/(r2-r1)
        psi_input=0.5*(r2-r1)*in_ovlp[input_channel,:]
        green=corner_green(E,ham,ovlp,n_emb,in_ovlp,out_ovlp)
        phi=-2.0*1j*sigma_input.imag*np.dot(green,psi_input)
        Psi=np.dot(phi,out_ovlp[output_channel,])
        output_current=-2.0*(np.conj(Psi)*Psi*sigma_output.imag).real
        T=output_current/input_current
    return T