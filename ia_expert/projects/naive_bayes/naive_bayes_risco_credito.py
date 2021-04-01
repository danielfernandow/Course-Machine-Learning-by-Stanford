#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 14:55:31 2021

@author: d5
"""

import pandas as pd

base = pd.read_csv('risco_credito.csv')

previsores = base.iloc[:, 0:4].values

classe = base.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()

previsores[:, 0] = label.fit_transform(previsores[:, 0])
previsores[:, 1] = label.fit_transform(previsores[:, 1])
previsores[:, 2] = label.fit_transform(previsores[:, 2])
previsores[:, 3] = label.fit_transform(previsores[:, 3])


from sklearn.naive_bayes import GaussianNB

classificador = GaussianNB()

#treinamento do algoritmo
classificador.fit(previsores, classe)

#historia boa, divida alta, garantia nenhuma, renda > 35
#previsao probabilidade
resultado = classificador.predict([[0, 0, 1, 2], [3, 0, 0, 0]])

print(classificador.classes_)
print(classificador.class_count_)
print(classificador.class_prior_)

