from __future__ import division
import numpy as np
import matplotlib.pyplot as pyplot
from scipy.optimize import curve_fit

# class const_Hubble(object):
#     '''
#     Docstring
#     '''
#
#     def __init__(self, distancias, velocidades_recesion, nosequemas):
#         # np.loadtxt('archivo_datos', usecols=[-2])
#         # np.loadtxt('archivo_datos', usecols=[-1])
#         # Los dos anteriores tengo que usarlos en mi codigo para cargar datos
#
#         self.distancias = distancias
#         self.veloc_recesion = velocidades_recesion

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


def sim_bootstrap (datos, H0_inicial):
    ''' Realiza la simulacion de bootstrap para encontrar un intervalo
    de confianza del 95%
    '''
    # Errores, simulacion de bootstrap:
    # N = numero de datos ; Nboot = int(N * np.log10(N)**2);
    # Pero N es muy pequeño (24 o 36)
    D, v = datos
    N = len(D) * 1000 # Para agrandar el N
    Nboot = int(N * np.log10(N)**2)

    D_boot[i] = np.zeros(Nboot)
    D_boot[i] = np.zeros(Nboot)
    H0_valores_1 = np.zeros(Nboot)
    H0_cov_1 = np.zeros(Nboot)
    H0_valores_2 = np.zeros(Nboot)
    H0_cov_2 = np.zeros(Nboot)
    for i in range(Nboot):
        s = np.random.randint(low=0, high=N, size=N)
        D_boot[i] = np.mean(D[s])
        v_boot[i] = np.mean(v[s])

        H0_valores_1[i], H0_cov_1[i] = curve_fit(func_minimizar_1,  D_boot[i],
                                           v_boot[i], H0_inicial)
        H0_valores_2[i], H0_cov_2[i] = curve_fit(func_minimizar_2,  v_boot[i],
                                           D_boot[i], H0_inicial)

    H0_prom = (h0_valores_1 + h0_valores_2) / 2

    # Ordenar los datos para encontrar los intervalos de confianza
    H0_1_sort = np.sort(H0_valores_1)
    H0_2_sort = np.sort(H0_valores_2)
    H0_prom_sort = np.sort(H0_prom)

    # Intervalo de confianza del modelo 1:
    limite_bajo_1 = H0_1_sort[int(Nboot * 0.025)]
    limite_alto_1 = H0_1_sort[int(Nboot * 0.975)]
    mod_1 = [H0_valores_1, limite_bajo_1, limite_alto_1]  # Valor a retornar

    # Intervalo de confianza del modelo 2:
    limite_bajo_2 = H0_2_sort[int(Nboot * 0.025)]
    limite_alto_2 = H0_2_sort[int(Nboot * 0.975)]
    mod_2 = [H0_valores_2, limite_bajo_2, limite_alto_2]  # Valor a retornar

    # Intervalo de confianza del modelo promedio:
    limite_bajo_p = H0_prom_sort[int(Nboot * 0.025)]
    limite_alto_p = H0_prom_sort[int(Nboot * 0.975)]
    mod_p = [H0_prom, limite_bajo_p, limite_alto_p]  # Valor a retornar

    return [mod_1, mod_2, mod_p]


# Main

# np.loadtxt('archivo_datos', usecols=[-2])
# np.loadtxt('archivo_datos', usecols=[-1])
