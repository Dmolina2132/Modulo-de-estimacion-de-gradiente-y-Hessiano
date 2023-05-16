# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 20:33:54 2023

@author: diego
"""

# Aquí juntamos todas las funciones necesarias para hacer la estimación de gradiente y Hessiano en 2 y 3 D
# Importamos los módulos necesarios
import numpy as np
import math as mt
import matplotlib.pyplot as plt
import scipy.optimize as spo
import random

# =============================================================================
# Funcion sigma ideal (no es obligatoria, pero junto con un mínimo N es suficiente para la convergencia)
# =============================================================================
# Para no crear funciones que constantemente tomen los 4 argumentos para la sigma la S, p y q
# hay que definirlas fuera de la función si es que se quiere usar la función de la forma especificada


def sigma(r):
    # Vamos a utilizar S y p predefinidos
    return (np.dot(np.dot(r, S), np.transpose(r))+np.dot(np.transpose(p), r)+q)[0]


def sigma_ruido(r):
    ruido = random.gauss(0, 1e-05)
    return sigma(r)+ruido

# =============================================================================
# Gradiente 2D
# =============================================================================

# Definimos ahora la función que nos genera la formación tomando el centro como argumento


def formacion_grad2D(c, N, D):
    pos = []
    for i in range(N):
        phi = 2*mt.pi*i/N
        rx = D*mt.cos(phi)
        ry = D*mt.sin(phi)
        ri = np.array([rx, ry], dtype=float)
        pos.append(c+ri)
    return pos


def estim_grad2D(sigma, pos, c, N, D):
    suma = 0
    for i in range(N):
        suma += sigma(pos[i])*(pos[i]-c)
    res = suma*2/(N*D**2)
    return res


# =============================================================================
# Gradiente 3D
# =============================================================================
def formacion_grad3D(c, n, D):
    N = 2*n
    pos = []
    # utilizamos la definición de theta que nos dan
    theta_F = mt.asin(mt.sqrt(2/3))
    theta_ipar = mt.pi-theta_F
    for i in range(0, N-1, 2):
        phi = 2*mt.pi*i/N
        vdir = np.array([[mt.sin(theta_ipar)*mt.cos(phi)],
                        [mt.sin(theta_ipar)*mt.sin(phi)], [mt.cos(theta_ipar)]])
        pos.append(c+D*vdir)
    for i in range(1, N, 2):
        phi = 2*mt.pi*i/N
        vdir = np.array([[mt.sin(theta_F)*mt.cos(phi)],
                        [mt.sin(theta_F)*mt.sin(phi)], [mt.cos(theta_F)]])
        pos.append(c+D*vdir)
    return pos


def estim_grad3D(sigma, pos, c, n, D):
    suma = 0
    N = 2*n
    for i in range(N):
        suma += sigma(pos[i])*(pos[i]-c)
    suma *= 3/(N*D**2)
    return suma

# =============================================================================
# Hessiano 2D
# =============================================================================


def formacion_H2D(c, N, D):
    pos = []
    for i in range(N):
        phi = 2*mt.pi*i/N
        rx = D*mt.cos(phi)
        ry = D*mt.sin(phi)
        ri = np.array([rx, ry], dtype=float)
        pos.append(c+ri)
    return pos


def calculo_K2D(N, sigma, pos, D, c):
    K = 0
    pos_arr = np.array(pos)
    for i in range(0, N):
        K += (sigma(pos[i])-sigma(c))*np.outer(pos_arr[i]-c, pos_arr[i]-c)

    K *= 16/(N*D**4)

    return K

# Vamos a usar el método de punto fijo para resolver la ecuación no lineal


def ec_H2D(H, R, K):
    # Definimos la función despejando H
    return (K - R@H@R.T)/3


def calculo_H_pf2D(ec, H0, tol, R, K):
    error = tol+1
    tol = 50*np.finfo(float).eps
    while error > tol:
        H = ec(H0, R, K)
        error = np.linalg.norm(H-H0, ord=2)
        # print(error)
        H0 = H
    return H

# =============================================================================
# Hessiano 3D
# =============================================================================


def formacion_H3D(c, n, D):
    N = 2*n
    pos = []
    # utilizamos la definición de theta que nos dan
    costheta = mt.sqrt(1/3)
    for i in range(0, N-1, 2):
        phi = 2*mt.pi*i/N
        vdir = np.array([[mt.sqrt(2/3)*mt.cos(phi)],
                        [mt.sqrt(2/3)*mt.sin(phi)], [costheta]])
        pos.append(c+D*vdir)
    for j in range(1, N, 2):
        phi = 2*mt.pi*j/N
        vdir = np.array([[mt.sqrt(2/3)*mt.cos(phi)],
                        [mt.sqrt(2/3)*mt.sin(phi)], [-costheta]])
        pos.append(c+D*vdir)
    pos.append(c+D*np.array([[0], [0], [1]]))
    pos.append(c+D*np.array([[0], [0], [-1]]))
    return pos


# Ahora toca toda la parte del cálculo del hessiano
# De nuevo tenemos que calcular la K primero
def calculo_K3D(n, sigma, pos, D, c):
    K = 0
    N = 2*n
    pos_arr = np.array(pos)
    for i in range(0, N+2):
        K += (sigma(pos[i])-sigma(c))*np.outer(pos_arr[i]-c, pos_arr[i]-c)

    K *= 18/(N*D**4)

    return K

# Cada elemento de K será el resultado de una ecuación para elementos del hessiano que dbemos resolver
# Planteamos el sistema:


def calculo_H3D(K, n):
    N = 2*n
    # Escribimos el sistema en forma matricial
    A = np.array([[3/2, 1/2, 1],
                  [1/2, 3/2, 1],
                  [1, 1, 1+18/N]])
    b = np.diag(K)
    # Sacamos la diagonal de H
    diagH = np.linalg.solve(A, b)
    # La metemos en una matriz
    H = np.diag(diagH)
    # Añadimos los elementos restantes
    H[1, 0] = H[0, 1] = K[1, 0]
    H[2, 0] = H[0, 2] = K[0, 2]/2
    H[1, 2] = H[2, 1] = K[1, 2]/2

    return H

# =============================================================================
# Algoritmo BFGS
# =============================================================================


def calculo_estH(grad_prev, grad_next, cprev, cnext, H0_inv, dim):
    s = cnext-cprev
    y = grad_next-grad_prev
    return(H0_inv +(s.T@y+y.T@H0_inv@y)*(s@s.T)/(s.T@y)**2-(H0_inv@y@s.T+s@y.T@H0_inv)/(s.T@y))
