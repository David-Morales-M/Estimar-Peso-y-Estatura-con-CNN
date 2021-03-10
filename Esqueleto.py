# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 09:08:25 2021

@author: David Morales
"""

import matplotlib
import random
import json
import numpy as np
import cv2


def crear_colores(n):
    # Seleccionando un estilo de mapa de color
    cmap = matplotlib.cm.get_cmap('gist_ncar')
    # Arreglo de colores 
    colors = []

    # Seleccionando colores al azar
    for i in range(n):
        colors.append(cmap(random.random()))
        
    return colors

# Método para dibujar el "esqueleto" o articulaciones
def dibujar_esqueleto(mask_img, joint_pos, colors):
    
    # Lista de vecinos entre extremidades 
    neighbors = {
        0: [1,14,15], 
        1: [2,5,8,11], 
        2: [3], 
        3: [4], 
        5: [6], 
        6: [7], 
        8: [9], 
        9: [10], 
        11: [12],
        12: [13], 
        14: [16], 
        15: [17]
    }
    
    # Para cada posición en el arreglo de posiciones de articulaciones
    for pos in joint_pos:
        cl = 0
        
        # Para cada punto en los vecinos
        for point in neighbors:
            if pos[point] != (0,0):
                for neighbor in neighbors[point]:
                    if pos[neighbor] != (0,0):
                        #Trazo de lineas que unen a los vecinos, en la Máscara
                        cv2.line(mask_img, pos[point][::-1], pos[neighbor][::-1], colors[cl], 2)
                        cl += 1

    return mask_img