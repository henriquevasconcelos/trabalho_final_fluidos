import numpy as np
import inputs as i
from scipy.optimize import fsolve

def haaland(f, D, Re, ϵ):
    F = (1/np.sqrt(f)) + 1.8*np.log((((ϵ/D)/3.7)**1.11) + (6.9/Re))
    return F

def colebrook(f, D, Re, ϵ):
    F = (1/np.sqrt(f)) + 2*np.log10(((ϵ/D)/3.7) + 2.51/(Re*np.sqrt(f)))
    return F

def calc_f(D, Re, ϵ, method=colebrook):
    f_guess_array = [0.006, 0.01, 0.015, 0.02, 0.05, 0.1]
    for f_guess in f_guess_array:
        sol = fsolve(method, f_guess, args=(D, Re, ϵ), full_output=True)
        if sol[2]==1:
            return sol[0][0]

def head_loss_12(Q, D, L, Leqcot, Leqvalv,ρ, mu, ϵ, return_velocity=False):
    v = (4*Q)/(np.pi*(D**2)) 
    Re = ρ*v*D/mu
    
    f = calc_f(D, Re, ϵ)
    
    hl = f * (L/D) * ((v**2)/2)
    hm = 8 * f * Leqcot * ((v**2)/2) + (1 * f * Leqvalv * ((v**2)/2))
    
    h_12 = hl+hm

    if return_velocity:
        return h_12, v
    
    else:
        return h_12

def head_loss_23A(Q, D, L, Leqcot, Leqvalv, ρ, mu, ϵ, return_velocity=False):
    Q_A = 0.6*Q
    
    v = (4*Q_A)/(np.pi*(D**2)) 
    Re = ρ*v*D/mu
    
    f = calc_f(D, Re, ϵ)
    
    hl = f * (L/D) * ((v**2)/2)
    hm = (15 * f * Leqcot * ((v**2)/2)) + (2 * f * Leqvalv * ((v**2)/2))
    
    h23A = hl+hm
    if return_velocity:
        return h23A, v 
    
    else:
        return h23A

def head_loss_23B(Q, D, L, Leqcot, Leqvalv, ρ, mu, ϵ, return_velocity=False):
    Q_B = 0.4*Q
    
    v = (4*Q_B)/(np.pi*(D**2)) 
    Re = ρ*v*D/mu
    
    f = calc_f(D, Re, ϵ)
    
    hl = f * (L/D) * ((v**2)/2)
    hm = (10 * f * Leqcot * ((v**2)/2)) + (3 * f * Leqvalv * ((v**2)/2))
    
    h23B = hl+hm
    if return_velocity:
        return h23B, v 
    
    else:
        return h23B

def head_loss_A(Q, D, L, Leqcurv, ρ, mu, ϵ, return_velocity=False):
    Q_A = 0.6*Q
    
    v = (4*Q_A)/(np.pi*(D**2)) 
    Re = ρ*v*D/mu
    
    f = calc_f(D, Re, ϵ)
    
    hl = f * (L/D) * ((v**2)/2)
    hm = 20 * f * Leqcurv * ((v**2)/2) 
    
    h_A = hl+hm

    if return_velocity:
        return h_A, v
    
    else:
        return h_A

def head_loss_B(Q, D, L, Leqcurv, ρ, mu, ϵ, return_velocity=False):
    Q_A = 0.4*Q
    
    v = (4*Q_A)/(np.pi*(D**2)) 
    Re = ρ*v*D/mu
    
    f = calc_f(D, Re, ϵ)
    
    hl = f * (L/D) * ((v**2)/2)
    hm = 20 * f * Leqcurv * ((v**2)/2) 
    
    h_B = hl+hm

    if return_velocity:
        return h_B, v
    
    else:
        return h_B


def head_loss_31(Q, D, L, Leqcot, Leqvalv, ρ, mu, ϵ, return_velocity=False):
    v = (4*Q)/(np.pi*(D**2)) 
    Re = ρ*v*D/mu
    
    f = calc_f(D, Re, ϵ)
    
    hl = f * (L/D) * ((v**2)/2)
    
    hm = 12 * f * Leqcot * ((v**2)/2) + (1 * f * Leqvalv * ((v**2)/2))
    
    K_tc = 30
    h_tc = K_tc * ((v**2)/2)
    
    h31 = hl+hm+h_tc 
    
    if return_velocity:
        return h31, v
    
    else:
        return h31

def calc_W_bomba(h_12, h_23A, h_A, h_31, ρ, Q):
    h_bomba = h_12 + h_23A + h_A + h_31
    P_bomba = h_bomba * ρ
    W_bomba = Q * P_bomba
    
    return W_bomba

def calc_C_total(D_array, L12, L23A, L23B, L31, b, F, C2, t, W_bomba, n, a, return_all_costs=False):
    C_P12 = ((4798.3 * D_array[0]) - 27.429) * L12
    C_P23A = ((4798.3 * D_array[1]) - 27.429) * L23A
    C_P23B = ((4798.3 * D_array[2]) - 27.429) * L23B
    C_P31 = ((4798.3 * D_array[3]) - 27.429) * L31
    C_P = C_P12 + C_P23A + C_P23B + C_P31
    C_PT = (a+b) * (1+F) * C_P
    C_OP = (C2 * t * W_bomba)/(n)
    C_total = C_PT + C_OP
   
    if return_all_costs:
        return C_PT, C_OP
        
    else:
        return C_total