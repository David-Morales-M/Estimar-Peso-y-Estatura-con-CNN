# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 18:24:02 2021

@author: David Morales
"""

import os              # Acceso al SO
import time            # Funciones del Tiempo
import sys             # Provee funcionalidades relacionadas con el interprete 
import numpy as np     # Estructura de Datos similar a Pandas 
from glob import glob  # Buscar el archivo en el Sistema
import torch           # Libreria para usar tensores
import cv2             # Análisis de Imagen
from torchvision import transforms # Transforma la imagen para el procesamiento
import matplotlib.pyplot as plt # Funciones de Matlab en Python
from Red_Peso import UNet # Arquitectura Unet de la Red_Peso
from Red_Estatura import UNet as HUNet # Arquitectura Unet de la Red_Estatura
import argparse  # Escritura de Interfaces (Argumentos - CMD)
from Esqueleto import crear_colores, dibujar_esqueleto # Esqueleto de la Segmentación Semántica
    
    #Comando para ejecutar el programa
    #python Estatura_y_Peso.py -i entrada/[nombre imagen] -g 0 -r 128


if __name__ == "__main__":
    
    # Configuración del Parser
    np.random.seed(23)
    # Creando un Menú de Ayuda
    parser = argparse.ArgumentParser(description="Estatura y Peso de Personas a partir de Imagenes")

    # Argumentos del Analizador
    parser.add_argument('-i', '--image', type=str, required=True, help='Directorio de Imagenes')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='Selección de GPU')
    parser.add_argument('-r', '--resolution', type=int, required=True, help='Resolución para imagen cuadrada')
    args = parser.parse_args()
    
    
    
    # Estatura
    # Invocación de la Red Convolucional de la Estatura 
    modelo_e = HUNet(128)
    # Carga el entrenamiento del modelo de la Estatura
    modelo_e_preentrenado = torch.load('models/model_ep_48.pth.tar')


    # Peso
    # Invocación de la Red Convolucional del Peso
    modelo_p = UNet(128, 32, 32)
    # Carga el entrenamiento del modelo del Peso
    modelo_p_preentrenado = torch.load('models/model_ep_37.pth.tar')
    
    # Asignando las capas a los tensores
    modelo_e.load_state_dict(modelo_e_preentrenado["state_dict"])
    modelo_p.load_state_dict(modelo_p_preentrenado["state_dict"])

    
    model = modelo_p
       
    # Leyendo la imagen
    assert ".jpg" in args.image or ".png" in args.image or ".jpeg" in args.image, "Por favor use una imagen con formato JPG o PNG"
    
    RES = args.resolution
    
    # Convirtiendo la imagen de BGR a RGB y en una matriz de flotantes32
    X = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB).astype('float32')
    # Escala de la imagen según las dimensiones de la matriz de la imagen
    escala = RES / max(X.shape[:2])
    
    # Imagen re-escalada con método Inter-Linear 
    # (Interpolación Bilinear) para asignar la ubicación de un pixel en la imagen
    X_escalado = cv2.resize(X, (0,0), fx=escala, fy=escala, interpolation=cv2.INTER_LINEAR) 
    
    # Si la imagen no tiene el mismo ancho y largo se re-escala
    # Rellena la matriz de la imagen con valores constantes
    if X_escalado.shape[1] > X_escalado.shape[0]:
        p_a = (RES - X_escalado.shape[0])//2
        p_b = (RES - X_escalado.shape[0])-p_a
        X = np.pad(X_escalado, [(p_a, p_b), (0, 0), (0,0)], mode='constant')
    elif X_escalado.shape[1] <= X_escalado.shape[0]:
        p_a = (RES - X_escalado.shape[1])//2
        p_b = (RES - X_escalado.shape[1])-p_a
        X = np.pad(X_escalado, [(0, 0), (p_a, p_b), (0,0)], mode='constant') 
    
    #Copiamos la imagen
    o_img = X.copy()
    # Transformación a valores de la matriz entre [0,1]
    X /= 255
    # Transformando la matriz a un tensor
    X = transforms.ToTensor()(X).unsqueeze(0)
        
    # Evaluación del Modelo
    model.eval()
    # Evitamos el cálculo del gradiente para reducir
    # uso computacional
    with torch.no_grad():
        # Creación de variables del peso para crear
        # la mascara, las articulaciones y el esqueleto
        m_p, j_p, _, w_p = model(X)
    
    del model
    
    model = modelo_e
        
    # Evaluación del Modelo
    model.eval()
    # Evitamos el cálculo del gradiente para reducir
    # uso computacional
    with torch.no_grad():
        # Creación de variables de la estatura para el
        # esqueleto y el calculo de estatura
        _, _, h_p = model(X)
    
    # Clasificando el formato de la imagen
    formato = '.png'
    if '.jpg' in args.image:
        formato = '.jpg'
    elif '.jpeg' in args.image:
        formato = '.jpeg'        
        
    # Obtención del valor máximo para la generación de 
    # la máscara y articulaciones y reducir una dimensión
    # en el modelo de entrada
    mascara_salida = m_p.argmax(1).squeeze().cpu().numpy()
    articulaciones_salida = j_p.argmax(1).squeeze().cpu().numpy()
    pred_2 = j_p.squeeze().cpu().numpy()
    
    # Etiquetas, Centroides, Estados en la Máscara
    n_etiquetas, etiquetas, estados, centroides = cv2.connectedComponentsWithStats(mascara_salida.astype('uint8'))
    # Método de esqueleto
    colores = crear_colores(30)
    # La matriz del esqueleto tendrá 3 dimensiones
    # con 128x128 pixeles
    esqueleto = np.zeros((128,128,3))
    
    # Arreglo para la posición de las articulaciones
    posicion_articulaciones = []
        
    
    for i in range(1, n_etiquetas):
        p_res = np.expand_dims((etiquetas==i).astype(int),0) * pred_2
        
        # Contador de posiciones
        ct_ = 1
        # Posiciones (Puntos de las articulaciones)
        posiciones = []
        # Usa como referencia 18 puntos en el cuerpo 
        for i in range(1,19):
            # Agregando las coordenadas de los puntos articulaciones a las posiciones
            posiciones.append(np.unravel_index(p_res[ct_,:,:].argmax(), p_res[ct_,:,:].shape))
            ct_ += 1
        
        # Agregando a todas las posiciones de las articulaciones
        posicion_articulaciones.append(posiciones)
    
    # Generación dela máscara en RGB (Amarillo)
    mascara_salida_RGB = np.concatenate([255*mascara_salida[:, :, np.newaxis],
                                  255*mascara_salida[:, :, np.newaxis],
                                  mascara_salida[:, :, np.newaxis],
                                  ], axis=-1)
    
    # Composición de la capa de Imagen Original y Máscara, condiferentes transparencias
    capa = cv2.addWeighted(o_img.astype('uint8'), 0.55, mascara_salida_RGB.astype('uint8'), 0.45, 0)
    # Creación del "esqueleto" de la persona sobre la capa anterior
    esqueleto = dibujar_esqueleto(capa/255, posicion_articulaciones, colores)

    # Creación de los archivos de salida (mascara, articulaciones, esqueleto)
    nombre_salida = args.image.split("/")[-1].replace(formato, '.mask.png')
    nombre_salida_articulaciones = args.image.split("/")[-1].replace(formato, '.joint.png')
    nombre_salida_esqueleto = args.image.split("/")[-1].replace(formato, '.skeleton.png')
    
    
    #Calculo del IMC
    altura = h_p.item()
    peso = 100*w_p.item()
    imc = (peso/(altura**2))
    
    if imc>30:
        estado = 'Obeso'
    elif imc > 25:
        estado = 'Sobrepeso'
    elif imc > 18:
        estado = 'Normal'
    else :
        estado = 'Bajo Peso'
            
    # Creación del archivo con la información de las medidas
    with open("salida/" + args.image.split("/")[-1].replace(formato, '.info.txt'), 'w') as out_file:
        out_file.write("Imagen: " + args.image)
        out_file.write("\nAltura: {:.1f} cm\nPeso: {:.1f} kg".format(100*h_p.item(), 100*w_p.item()))
        out_file.write("\nIMC: {:.2f} cm".format(imc))
        out_file.write("\nEstado:" + estado)
    
    # Guardado de la máscara
    cv2.imwrite("salida/" + nombre_salida, (255*mascara_salida).astype('uint8'))
    # Guarda una matriz como imagen con mapa de color
    plt.imsave("salida/" + nombre_salida_articulaciones, articulaciones_salida, cmap='jet')
    plt.imsave("salida/" + nombre_salida_esqueleto, esqueleto)
    
    # Impresión de los datos en consola
    print("\nImagen: " + args.image)
    print("Altura: {:.1f} cm\nPeso: {:.1f} kg".format(100*h_p.item(), 100*w_p.item()))
    print("\nIMC: {:.2f} cm".format(imc))
    print("Estado:" + estado)
    print("Máscaras y Articulaciones generadas en el directorio /salida")
    print("Los Datos han sido guardados exitosamente en un archivo .txt en el directorio /salida")
        
    del model