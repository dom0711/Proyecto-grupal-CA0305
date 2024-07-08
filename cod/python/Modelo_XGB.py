# -*- coding: utf-8 -*-
"""
Modulo Modelo_XGB

"""

import xgboost as xgb
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
import numpy as np
from Preproceso import Preproceso 

class ModeloXGBoost:
    def __init__(self, data):
        self.data = data
        self.X = self.data.drop(columns=['isFraud'])
        self.Y = self.data['isFraud']
        self.pipeline = None
        self.best_params = {
            'learning_rate': 0.1,
            'max_depth': 5,
            'n_estimators': 100,
            'subsample': 1.0,
            'n_jobs': -1  # Paralelización
        }
        self.preproceso_instancia = Preproceso()

    def crear_pipeline(self):
        preproceso = self.preproceso_instancia.preprocesar_datos()
        self.pipeline = Pipeline([
            ('preproceso', preproceso),
            ('classifier', xgb.XGBClassifier(**self.best_params))
        ])
        return self.pipeline

    def entrenar_modelo(self):
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=0.25, random_state=7, 
                                                            stratify=self.Y)
        self.crear_pipeline()
        self.pipeline.fit(X_train, Y_train)
        Y_predict_proba = self.pipeline.predict_proba(X_test)[:, 1]
        roc_auc_test = roc_auc_score(Y_test, Y_predict_proba)
        print("ROC AUC del conjunto de prueba:", roc_auc_test)
        return roc_auc_test

    def validacion_cruzada(self, num_estratos=5):
        self.crear_pipeline()
        estratos_kfold = StratifiedKFold(n_splits=num_estratos, shuffle=True, random_state=7)
        roc_auc_vector = cross_val_score(self.pipeline, self.X, self.Y, cv=estratos_kfold, scoring='roc_auc', n_jobs=-1)
        roc_auc_promedio = np.mean(roc_auc_vector)
        print("ROC AUC promedio utilizando validación cruzada estratificada:", roc_auc_promedio)
        return roc_auc_promedio