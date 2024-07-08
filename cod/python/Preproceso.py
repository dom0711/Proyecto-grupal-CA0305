# -*- coding: utf-8 -*-
"""
Modulo Preproceso

"""

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class Preproceso:
    def __init__(self):
        '''
        Método constructor de la clase

        Returns:
            Un objeto del tipo Preproceso

        '''
        self.variables_numericas = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 
                                    'newbalanceDest', 'isFlaggedFraud']
        self.variables_categoricas = ['type']

    def preprocesar_datos(self):
        '''
        Método que se encarga de trabajar un poco los datos antes de aplicar el modelo

        Returns:
            preproceso: corresponde a los datos después de aplicarse el método, tipo pandas data frame

        '''
        reescalar = StandardScaler()
        codificar = OneHotEncoder(drop='first', sparse_output=False)
        preproceso = ColumnTransformer(
            transformers=[
                ('num', reescalar, self.variables_numericas),
                ('cat', codificar, self.variables_categoricas)
            ])
        return preproceso