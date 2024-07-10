# -*- coding: utf-8 -*-
"""
Modulo Data_Preproceso

"""

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

class DataPreproceso:
    
    def __init__(self, file_path):
        '''
        Método constructor de la clase. Crea dos listas con los dos tipos de variables independientes
        
        Parametros:
          Recibe el path de la base de datos
          
         Returns:
             Un objeto del tipo DataPreproceso
        '''
        self.file_path = file_path
        self.variables_numericas = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud']
        self.variables_categoricas = ['type']
    
    def __str__(self):
        '''
        Método str de la clase
        '''
        return f"Carga data:{self.file_path} y crea el preproceso"
    
    
    def cargar_data(self):
        '''
        Método que carga la data
        
        Returns:
            Un objeto tipo pandas data frame
        '''
        data = pd.read_csv(self.file_path)
        return data
    
    
    def preprocesar_datos(self):
        '''
        Método que crea el preproceso para el pipeline
        
        Returns:
            Un Columntransformer con dos diferentes transformadores para cada tipo de variable. 
            Uno para imputar datos nulos y uno para procesar mejor la data.
        '''
        reescalar = StandardScaler()
        codificar = OneHotEncoder(drop='first', sparse_output=False)
        imputar_num = SimpleImputer(strategy='mean')
        imputar_cat = SimpleImputer(strategy='most_frequent')

        preproceso = ColumnTransformer(
            transformers=[
                ('num', Pipeline([
                    ('imputar', imputar_num),
                    ('reescalar', reescalar)
                ]), self.variables_numericas),
                ('cat', Pipeline([
                    ('imputar', imputar_cat),
                    ('codificar', codificar)
                ]), self.variables_categoricas)
                ])
        return preproceso


    def cargar_data_y_preprocesar(self):
        '''
        Método que utiliza los dos primeros métodos para cargar la data y crear el columntransformer
        
        Returns:
            Un objeto tipo pandas data frame y el objeto preproceso
        
        '''
        data = self.cargar_data()
        preproceso = self.preprocesar_datos()
        return data, preproceso
    