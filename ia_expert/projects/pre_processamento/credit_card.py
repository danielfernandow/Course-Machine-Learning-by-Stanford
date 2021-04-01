#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 23:01:18 2021

@author: d5
"""

import pandas as pd
import numpy as np

base = pd.read_csv('credit_data.csv')

base.loc[base['age'] < 0]

# Tratamento valor inválido
### preencher os valores inválidos da idade, com sua média

base.mean()

base['age'][base.age > 0].mean()
#40.92

base.loc[base.age < 0, 'age'] = 40.92

# divisao de base

previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(previsores[:, 0:3])
previsores[:, 0:3] = imputer.transform(previsores[:, 0:3])

# Escolonamento

# Padronização (Standrdisation)
# Normalização (Normalization)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

previsores = scaler.fit_transform(previsores)