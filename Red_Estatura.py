# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 18:24:02 2021

@author: David Morales
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

#  Convolución hacia Arriba
class UpBlock(nn.Module):
    
    # Función que inicializa los módulos internos
    # in_ch, out_ch = Canales de entrada y salida (enteros)
    # mode = Para el muestreo de la imagen (Algoritmo de Escalamiento)
    def __init__(self, in_ch, out_ch, mode='bilinear'):
        super(UpBlock, self).__init__()

        # Aumento de Atributo Conv1 para la Red Neuronal Convolucional
        # Conv1 es un contenedor secuencial de módulos
        self.conv1 = nn.Sequential(
            # Genera el espacio a partir de un dato
            #// scale_factor = "2" | Multiplicador de Tamaño Espacio 
            #// mode = "Bilinear" | Para el muestreo de la imagen 
            nn.Upsample(scale_factor=2, mode=mode),
            # Convolución en 2D dada una señal compuesta por planos
            # in_ch = Canal de Entrada 
            # out_ch = Canal de Salida  
            # Tamaño del Kernel 3 
            # padding = Controla el número de 0's de relleno para completar la dimensión
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            # Función de Activación de Unidad de Rectificación Lineal
            nn.ReLU()
        )
        
        # Conv2 para la Red Neuronal Convolucional
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU()
        )

        # Función de C
        # x_up, x_down = Tensores
    def forward(self, x_up, x_down): 
        # Tensores de la Convolucion 1
        x_up = self.conv1(x_up)
        # Concatena la secuencia dada de tensores
        x_up = torch.cat([x_up, x_down], dim=1)
        # Tensores de la Convolucion 2
        x_up = self.conv2(x_up)

        return x_up
    
    
    # Clase del Bloque Final de Red Neuronal
class FinalBlock(nn.Module):
    
    # Función 
    # in_ch = Canal de Entrada
    # pool_size = Tamaño de Piscina, Reduce Datos pero
    # preserva informacion
    # h_channel = Canal de altura
    def __init__(self, in_ch, pool_size, h_channel):
        super(FinalBlock, self).__init__()
                
        # Convolución a las capas de máscara y articulaciones
        self.mask_out = nn.Conv2d(in_ch, 2, 1)
        self.joint_out = nn.Conv2d(in_ch, 19, 1)
        self.pool_size = pool_size
        
        # Primera Convolución
        self.height_1 = nn.Sequential(
            nn.Conv2d(in_ch, h_channel, 1),
            nn.ReLU()
        )
        
        # Segunda Convolución
        self.height_2 = nn.Sequential(
            # Aplicación de Transformación lineal, de una entrada a una salida 32*32
            nn.Linear(pool_size*pool_size*h_channel, 1024), # 1024
            # "Apaga" algunas neuronas para evitar el sobre-entrenamiento
            nn.Dropout(0.15), # 0.15
            nn.ReLU(),
            nn.Linear(1024, 1) # 1024
        )
        
    def forward(self, x):
        # Re-escala la dimensión del tensor de la máscara | 1D
        mask = torch.nn.Softmax(1)(self.mask_out(x))
        joint = self.joint_out(x)
        
        # Agrupa en promedio una señal de entrada
        height = F.adaptive_avg_pool2d(x, (self.pool_size, self.pool_size))
        height = self.height_1(height)
        # Copia el mismo tensor en una shape diferente
        height = height.view(height.size(0), -1)
        height = self.height_2(height)
        
        return mask, joint, height

    # Convolución hacia Abajo 
class DownBlock(nn.Module):
    
    def __init__(self, in_ch, out_ch):
        super(DownBlock, self).__init__()
                
        # Convolución
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.conv(x)


# Arquitectura de Red Convolucional    
class UNet(nn.Module):

    def __init__(self, min_neuron, pool_size=32, h_ch=32):
        super(UNet, self).__init__()
        
        self.conv_down1 = DownBlock(3, min_neuron)
        self.conv_down2 = DownBlock(min_neuron, 2*min_neuron)
        self.conv_down3 = DownBlock(2*min_neuron, 4*min_neuron)
        self.conv_down4 = DownBlock(4*min_neuron, 8*min_neuron)
        self.conv_down5 = DownBlock(8*min_neuron, 16*min_neuron)
        
        self.conv_upsample1 = UpBlock(16*min_neuron, 8*min_neuron)
        self.conv_upsample2 = UpBlock(8*min_neuron, 4*min_neuron)
        self.conv_upsample3 = UpBlock(4*min_neuron, 2*min_neuron)
        self.conv_upsample4 = UpBlock(2*min_neuron, min_neuron)

        self.conv_out = FinalBlock(min_neuron, pool_size, h_ch)
        
    def forward(self, x):
        #Baja
        conv1 = self.conv_down1(x)
        pool1 = nn.MaxPool2d(kernel_size=2, stride=2)(conv1)

        conv2 = self.conv_down2(pool1)
        pool2 = nn.MaxPool2d(kernel_size=2, stride=2)(conv2)
        
        conv3 = self.conv_down3(pool2)
        pool3 = nn.MaxPool2d(kernel_size=2, stride=2)(conv3)
        
        conv4 = self.conv_down4(pool3)
        pool4 = nn.MaxPool2d(kernel_size=2, stride=2)(conv4)
        
        conv5 = self.conv_down5(pool4)

        # Sube
        up6 = self.conv_upsample1(conv5, conv4)
        up7 = self.conv_upsample2(up6, conv3)
        up8 = self.conv_upsample3(up7, conv2)
        up9 = self.conv_upsample4(up8, conv1)
        
        # Procesa
        return self.conv_out(up9)