# -*- coding: utf-8 -*-
"""
Modulo Data

"""

import pandas as pd

class Data:
    
    # Constructor
    def __init__(self, file_path):
        '''
        Método constructor de la clase Data
        
        Parmetros:
            file_path: corresponde a la ruta donde está el acrhivo csv a cargar, tipo string

        Returns:
            Un objeto del tipo Data

        '''
        self.file_path = file_path
        self.data = None
        
    # Getters
    @property
    def file_path(self):
        '''
        Método Get del atributo file_path de la clase

        Returns:
            El atributo file_path de la clase
        '''
        return self.__file_path
    @property
    def data(self):
        '''
        Método Get del atributo data de la clase

        Returns:
            El atributo data de la clase
        '''
        return self.__data
    
    # Setters
    @file_path.setter 
    def file_path(self, file_path):
        '''
        Método Set del atributo file_path de la clase, asgina al atributo file_path el valor file_path

        Parametros:
            file_path: corresponde a la ruta donde está el acrhivo csv a cargar, tipo string
        '''
        self.__file_path = file_path
    @data.setter 
    def data(self, data):
        '''
        Método Set del atributo data de la clase, asgina al atributo data el valor data

        Parametros:
            data: corresponde al data frame cargado, tipo pandas data frame
        '''
        self.__data = data
        
        
        
    # Str
    def __str__(self):
        '''
        Método str de la clase
        
        Returns:
            Imprime los atributos del objeto en forma de string.
        '''
        return f''' Ruta del archivo: {self.__file_path} 
                \n Datos: {self.__data}
                '''
        
    def cargar_data(self):
        '''
        Método que se encarga de cargar la basa de datos

        Returns:
            self.data: el data frame que contienen los datos, tipo pandas data frame

        '''
        try:
            self.data = pd.read_csv(self.file_path)
            print(f"Data cargada correctamente {self.file_path}")
        except Exception as e:
            print(f"No se pudo cargar la data: {e}")
        return self.data