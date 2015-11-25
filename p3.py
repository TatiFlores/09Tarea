#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
'''
En este script se encuentra una linea recta que mejor modela la relación entre
el flujo de la banda i y la banda z, incluyendo los intervalos de confianza al
95% para los parámetros de la línea recta.
'''

# Datos
flujo_i = np.loadtxt("data/DR9Q.dat", usecols=[80]) * 3.631
error_i = np.loadtxt("data/DR9Q.dat", usecols=[81]) * 3.631
flujo_z = np.loadtxt("data/DR9Q.dat", usecols=[82]) * 3.631
error_z = np.loadtxt("data/DR9Q.dat", usecols=[83]) * 3.631

# ========================= Simulacion montecarlo ===========================
np.random.seed(800)
Nmc = 1000000
pendiente = np.zeros(Nmc)
coef_posic = np.zeros(Nmc)
for i in range(Nmc):
    r = np.random.normal(0, 1, size=len(flujo_i))  # len -> largo medicion
    muestra_i = flujo_i + error_i * r
    muestra_z = flujo_i + error_z * r
    pendiente[i], coef_posic[i] = np.polyfit(muestra_i, muestra_z, 1)

pendiente = np.sort(pendiente)
limite_bajo_pendiente = pendiente[int(Nmc * 0.025)]
limite_alto_pendiente = pendiente[int(Nmc * 0.975)]

coef_posic = np.sort(coef_posic)
limite_bajo_coef = coef_posic[int(Nmc * 0.025)]
limite_alto_coef = coef_posic[int(Nmc * 0.975)]

print """El intervalo de confianza al 95% para la pendiente
         es: [{}:{}]""".format(limite_bajo_pendiente, limite_alto_pendiente)
print ' '
print """El intervalo de confianza al 95% para el coeficiente de posicion
         es: [{}:{}]""".format(limite_bajo_coef, limite_alto_coef)
print ' '
# =========================== Ajuste lineal ===============================

PENDIENTE, COEF_POSIC = np.polyfit(flujo_i, flujo_z, 1)
print 'Pendiente', PENDIENTE
print 'Coeficiente de posicion', COEF_POSIC
x = np.linspace(min(flujo_i) - max(error_i), max(flujo_i) + max(error_i), 1000)
y = x * PENDIENTE + COEF_POSIC

# =============================== Plots ====================================

plt.figure(1, figsize=(17, 7))
plt.suptitle('Histogramas', fontsize=16)
plt.subplots_adjust(hspace=.5)
plt.subplot(121)
plt.xlabel('Pendientes')
plt.ylabel('Frecuencias')
plt.hist(pendiente, bins=40, normed=True, color='green')
plt.subplot(122)
plt.xlabel('Coeficientes de posici'u'ó''n')
plt.ylabel('Frecuencias')
plt.hist(coef_posic, bins=40, normed=True)
plt.savefig('p3_hist.eps')

plt.figure(2)
plt.clf()
plt.errorbar(flujo_i, flujo_z, xerr=error_i, yerr=error_z, fmt='mo',
             label='Datos')
plt.plot(x, y, 'c', label='Ajuste lineal', lw=2)
plt.xlabel('Flujo banda i $[10^{-6} Jy]$')
plt.ylabel('Flujo banda z $[10^{-6} Jy]$')
plt.legend(loc='upper left')
plt.xlim(-10, 500)
plt.ylim(-300, 600)
plt.savefig('p3_ajuste.eps')

plt.show()
