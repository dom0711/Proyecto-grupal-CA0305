# -*- coding: utf-8 -*-
"""
Modulo Modelo_Logistico

"""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc
import numpy as np

class ModeloLogistico:
    
    def __init__(self, data, preproceso):
        '''
        Método constructor de la clase
        
        Parametros:
          Recibe la base datos y el preproceso
          
         Returns:
             Un objeto del tipo ModeloLogistico
        '''
        self.data = data
        self.preproceso = preproceso
        self.pipeline = None
    
    def __str__(self):
        '''
        Método str de la clase
        '''
        return "ModeloLogistico con LogisticRegression"
    
    
    def crear_pipeline(self):
        '''
        Método que crea el pipeline con el preproceso y la regresion logistica
        '''
        classifier = LogisticRegression()
        self.pipeline = Pipeline([
            ('preproceso', self.preproceso),
            ('classifier', classifier)
        ])
      
    
    def entrenar_modelo(self):
        '''
        Método que entrena el modelo, calcula las probabilidades y calcula el ROC AUC para el conjunto 
        de prueba
        
        Returns:
            La metrica ROC AUC
        '''
        X,Y = self.data.drop(columns=['isFraud']), self.data['isFraud']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=7, stratify=Y)
        self.pipeline.fit(X_train, Y_train)
        Y_predict_proba = self.pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(Y_test, Y_predict_proba)
        roc_auc_test = auc(fpr, tpr)
        print("ROC AUC del conjunto de prueba:", roc_auc_score(Y_test, Y_predict_proba))
        return roc_auc_test, fpr, tpr
    
    
    def validacion_cruzada(self, num_estratos=5):
        '''
        Método que se encarga de realizar la validacion cruzada estratificada
        
        Returns:
            El promedio de los ROC AUC de las distintas iteraciones
        '''
        X,Y = self.data.drop(columns=['isFraud']), self.data['isFraud']
        estratos_kfold = StratifiedKFold(n_splits=num_estratos, shuffle=True, random_state=7)
        roc_auc_vector = cross_val_score(self.pipeline, X, Y, cv=estratos_kfold, scoring='roc_auc', n_jobs=4)
        roc_auc_promedio = np.mean(roc_auc_vector)
        print("ROC AUC promedio utilizando validación cruzada estratificada:", roc_auc_promedio)
        return roc_auc_promedio
