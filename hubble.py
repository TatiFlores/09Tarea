#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def modelo_1(parametro, D):
    '''
    El modelo es de la forma v = H0 * D, donde:
        - v : velocidad de recesion
        - D : distancia
        - H0: Constante de Hubble
    '''
    H0 = parametro
    v = H0 * D
    return v


def func_minimizar_1(datos_distancia, parametro):
    D = datos_distancia
    return modelo_1(parametro, D)


def modelo_2(parametro, v):
    '''
    El modelo es de la forma v / H0 = D, donde:
        - v : velocidad de recesion
        - D : distancia
        - H0: Constante de Hubble
    '''
    H0 = parametro
    D = v / H0
    return D


def func_minimizar_2(datos_velocidad, parametro):
    v = datos_velocidad
    return modelo_2(parametro, v)


def sim_bootstrap(datos, H0_inicial):
    ''' Realiza la simulacion de bootstrap para encontrar un intervalo
    de confianza del 95%
    '''
    # Errores, simulacion de bootstrap:
    # N = numero de datos ; Nboot = int(N * np.log10(N)**2);
    # Pero N es muy pequeño (24 o 36)
    D, v = datos
    N = len(D)
    Nboot = int(N * np.log10(N)**2) * 100  # Aqui agrando Nboot (*100)

    np.random.seed(800)
    H0_valores = np.zeros(Nboot)
    for i in range(Nboot):
        s = np.random.randint(low=0, high=N, size=N)
        D_boot = D[s]
        v_boot = v[s]
        H0_1, H0_cov_1 = curve_fit(func_minimizar_1, D_boot, v_boot,
                                   H0_inicial)
        H0_2, H0_cov_2 = curve_fit(func_minimizar_2, v_boot, D_boot,
                                   H0_inicial)
        H0_bisec = biseccion(H0_1, H0_2)
        H0_valores[i] = H0_bisec

    H0_sort = np.sort(H0_valores)

    # Intervalo de confianza:
    limite_bajo = H0_sort[int(Nboot * 0.025)]
    limite_alto = H0_sort[int(Nboot * 0.975)]
    return [H0_valores, limite_bajo, limite_alto]


def biseccion(parametro1, parametro2):
    b1 = parametro1
    b2 = parametro2
    b_biseccion = (b1 * b2 - 1 + np.sqrt((1 + b1**2) *
                                         (1 + b2**2))) / (b1 + b2)
    return b_biseccion

# Main

'''
------------------------------ Pregunta 1 -------------------------------------
'''

distance = np.loadtxt("data/hubble_original.dat", usecols=[-2])
recession_velocity = np.loadtxt("data/hubble_original.dat", usecols=[-1])


# =================== Minimizar Chi cuadrado ===============================
H0_inicial = 10
D = np.linspace(-0.5, 2.5, 100)
v = np.linspace(-400, 1200, 100)

# Modelo 1:
H0_optimo_mod1, H0_cov_mod1 = curve_fit(func_minimizar_1, distance,
                                        recession_velocity, H0_inicial)

# Modelo 2:
H0_optimo_mod2, H0_cov_mod2 = curve_fit(func_minimizar_2, recession_velocity,
                                        distance, H0_inicial)

# Modelo promedio:
H0_optimo_modprom = biseccion(H0_optimo_mod1, H0_optimo_mod2)

print 'PREGUNTA 1:'
print '- H0 modelo 1 = ', H0_optimo_mod1[0], '[km / s / Mpc]'
print '- H0 modelo 2 = ', H0_optimo_mod2[0], '[km / s / Mpc]'
print '- H0 modelo promedio = ', H0_optimo_modprom[0], '[km / s / Mpc]'
# H0_optimo ... [0] => [0] es porque H0 es un arreglo y quiero solo el numero


# ==================== Intervalo de confianza =============================
datos = [distance, recession_velocity]
BOOTSTRAP = sim_bootstrap(datos, H0_inicial)
H0 = BOOTSTRAP[0]
limite_bajo = BOOTSTRAP[1]
limite_alto = BOOTSTRAP[2]
print "El intervalo de confianza al 95% es: [{}, {}]".format(limite_bajo,
                                                             limite_alto)
print ' '

# =============================== Plots ====================================

plt.figure(1)
plt.clf()
plt.plot(distance, recession_velocity, 'm^', label='Datos')
plt.plot(D, func_minimizar_1(D, H0_optimo_mod1), 'limegreen',
         label='Modelo $v = H_0 * D$')
plt.plot(func_minimizar_2(v, H0_optimo_mod2), v, 'mediumblue',
         label='Modelo $D = v / H_0$')
plt.plot(D, func_minimizar_1(D, H0_optimo_modprom), 'orange',
         label='Modelo promedio')
plt.xlabel('Distancia [Mpc]', fontsize=14)
plt.ylabel('Velocidad de recesion [km / s]', fontsize=14)
plt.xlim(-0.5, 2.5)
plt.grid(False)
plt.legend(loc='upper left')

plt.figure(2)
plt.clf()
plt.hist(H0, bins=30)
plt.xlabel('Valor $H_0 [km \cdot s^{-1}\cdot Mpc^{-1}]$', fontsize=14)
plt.ylabel('Frecuencia', fontsize=14)
plt.axvline(np.mean(H0), color='r')

'''
------------------------------ Pregunta 2 -------------------------------------
'''

distance = np.loadtxt("data/SNIa.dat", usecols=[-2])
recession_velocity = np.loadtxt("data/SNIa.dat", usecols=[-1])


# =================== Minimizar Chi cuadrado ===============================
H0_inicial = 4
D = np.linspace(4000, 31000, 100)
v = np.linspace(40, 500, 100)

# Modelo 1:
H0_optimo_mod1, H0_cov_mod1 = curve_fit(func_minimizar_1, distance,
                                        recession_velocity, H0_inicial)

# Modelo 2:
H0_optimo_mod2, H0_cov_mod2 = curve_fit(func_minimizar_2, recession_velocity,
                                        distance, H0_inicial)

# Modelo promedio:
H0_optimo_modprom = biseccion(H0_optimo_mod1, H0_optimo_mod2)

print 'PREGUNTA 2:'
print '- H0 modelo 1 = ', H0_optimo_mod1[0], '[km / s / Mpc]'
print '- H0 modelo 2 = ', H0_optimo_mod2[0], '[km / s / Mpc]'
print '- H0 modelo promedio = ', H0_optimo_modprom[0], '[km / s / Mpc]'


# ==================== Intervalo de confianza =============================
datos = [distance, recession_velocity]
BOOTSTRAP = sim_bootstrap(datos, H0_inicial)
H0 = BOOTSTRAP[0]
limite_bajo = BOOTSTRAP[1]
limite_alto = BOOTSTRAP[2]
print "El intervalo de confianza al 95% es: [{}, {}]".format(limite_bajo,
                                                             limite_alto)
# =============================== Plots ====================================

plt.figure(3)
plt.clf()
plt.plot(distance, recession_velocity, 'm^', label='Datos')
plt.plot(D, func_minimizar_1(D, H0_optimo_mod1), 'limegreen',
         label='Modelo $v = H_0 * D$')
plt.plot(func_minimizar_2(v, H0_optimo_mod2), v, 'mediumblue',
         label='Modelo $D = v / H_0$')
plt.plot(D, func_minimizar_1(D, H0_optimo_modprom), 'orange',
         label='Modelo promedio')
plt.xlabel('Distancia [Mpc]', fontsize=14)
plt.ylabel('Velocidad de recesion [km / s]', fontsize=14)
plt.grid(False)
plt.legend(loc='upper left')

plt.figure(2)
plt.clf()
plt.hist(H0, bins=30)
plt.xlabel('Valor $H_0 [km \cdot s^{-1}\cdot Mpc^{-1}]$', fontsize=14)
plt.ylabel('Frecuencia', fontsize=14)
plt.axvline(np.mean(H0), color='r')
plt.show()
