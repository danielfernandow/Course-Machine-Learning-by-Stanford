#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 11:01:47 2021

@author: d5
"""

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

for i in range(0, 4):
    previsores[:, i] = label.fit_transform(previsores[:, i])
"""previsores[:, 0] = label.fit_transform(previsores[:, 0])
previsores[:, 1] = label.fit_transform(previsores[:, 1])
previsores[:, 2] = label.fit_transform(previsores[:, 2])
previsores[:, 3] = label.fit_transform(previsores[:, 3])"""


from sklearn.tree import DecisionTreeClassifier, export

classificador = DecisionTreeClassifier()

#treinamento do algoritmo
classificador.fit(previsores, classe)
print(classificador.feature_importances_)

export.export_graphviz(classificador, out_file='arvore.dot', 
                       feature_names=['historia', 'divida', 'garantia', 'renda'],
                       class_names=['alto', 'moderado', 'baixo'],
                       filled = True, leaves_parallel=True)

#historia boa, divida alta, garantia nenhuma, renda > 35
#previsao probabilidade
resultado = classificador.predict([[0, 0, 1, 2], [3, 0, 0, 0]])

print(classificador.classes_)


