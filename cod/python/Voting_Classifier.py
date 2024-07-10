# -*- coding: utf-8 -*-
"""
Modulo Voting_Classifier

"""

from sklearn.ensemble import VotingClassifier
from Modelo_XGB import ModeloXGBoost
from Modelo_Logistico import ModeloLogistico
from Modelo_Random_Forest import ModeloRandomForest
from Modelo_Lightgbm import ModeloLightGBM
from Modelo_SVC import ModeloSVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
import numpy as np

class VotingClassifierModel:
    
    
    def __init__(self, data, preproceso):
        '''
        Método constructor de la clase
        
        Parametros:
            Recibe la base datos y el preproceso
            
        Returns:
            Un objeto de tipo VotingClassifierModel
        '''
        self.data = data
        self.preproceso = preproceso
        self._X = self.data.drop(columns=['isFraud'])
        self._Y = self.data['isFraud']
        self.pipeline = None

    def __str__(self):
        '''
        Método str de la clase
        '''
        return "VotingClassifierModel con múltiples estimators"
    
    def crear_modelos_base(self):
        '''
        Método que crea los modelos base para el voting classifier
        
        Returns:
            Los distintos modelos base
        '''
        modelo_xgb = ModeloXGBoost(self.data, self.preproceso)
        modelo_logistico = ModeloLogistico(self.data, self.preproceso)
        modelo_rf = ModeloRandomForest(self.data, self.preproceso)
        modelo_lgb = ModeloLightGBM(self.data, self.preproceso)
        modelo_svc = ModeloSVC(self.data, self.preproceso)
        
        modelo_xgb.crear_pipeline()
        modelo_logistico.crear_pipeline()
        modelo_rf.crear_pipeline()
        modelo_lgb.crear_pipeline()
        modelo_svc.crear_pipeline()
        
        return [
            ('xgb', modelo_xgb.pipeline),
            ('logistic', modelo_logistico.pipeline),
            ('rf', modelo_rf.pipeline),
            ('lgb', modelo_lgb.pipeline),
            ('svc', modelo_svc.pipeline)
        ]
    
    def crear_voting_classifier(self):
        '''
        Método que crea el voting classifier. 
        
        Returns:
            Crea el voting classifier. Los modelos base son pipelines
        '''
        base_estimators = self.crear_modelos_base()
        self.pipeline = VotingClassifier(estimators=base_estimators, voting='soft', n_jobs=4)
        

    def entrenar_voting_classifier(self):
        '''
        Método entrena el modelo ensamblado, calcula las probabilidades y calcula el ROC AUC para el 
        conjunto de prueba
        
        Returns:
            La metrica ROC AUC
        '''
        X,Y = self.data.drop(columns=['isFraud']), self.data['isFraud']
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, 
                                                            random_state=7, stratify=Y)
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
        roc_auc_vector = cross_val_score(self.pipeline, X, Y, cv=estratos_kfold, 
                                         scoring='roc_auc', n_jobs=4)
        roc_auc_promedio = np.mean(roc_auc_vector)
        print("ROC AUC promedio utilizando validación cruzada estratificada:", roc_auc_promedio)
        return roc_auc_promedio

    

