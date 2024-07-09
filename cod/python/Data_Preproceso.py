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
        self.file_path = file_path
        self.variables_numericas = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud']
        self.variables_categoricas = ['type']

    def cargar_data(self):
        data = pd.read_csv(self.file_path)
        return data

    def preprocesar_datos(self):
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
        data = self.cargar_data()
        preproceso = self.preprocesar_datos()
        return data, preproceso