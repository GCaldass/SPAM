# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 18:11:33 2021

@author: alexa
"""

import pandas as pd
import time

# Importar Ficheiro
email_data = pd.read_csv('spam.csv', header=None, sep='\t', names=['Label', 'E-Mail'],
                         encoding='cp1252')

# Dados Raw
print('Dados Raw')
print(email_data.head())

print('\nDivisão por Labels')
print(email_data.groupby('Label').count())

# Tempo
tic = time.perf_counter()

# Tratamento dos dados
dados_limpos_email = email_data.copy()
dados_limpos_email['E-Mail'] = dados_limpos_email['E-Mail'].str.replace('\W+', ' ').str.replace('\s+', ' ').str.strip()
dados_limpos_email['E-Mail'] = dados_limpos_email['E-Mail'].str.lower()
dados_limpos_email['E-Mail'] = dados_limpos_email['E-Mail'].str.split()

# Dados Limpos
print('\nDados Limpos')
print(dados_limpos_email['E-Mail'].head())

# Percentagem dos Dados
print(dados_limpos_email['Label'].value_counts() / email_data.shape[0] * 100)

# Divisão dos Dados

treino_data = dados_limpos_email.sample(frac=0.8, random_state=1).reset_index(drop=True)
teste_data = dados_limpos_email.drop(treino_data.index).reset_index(drop=True)
treino_data = treino_data.reset_index(drop=True)

# treino_data
print('\nCarateristicas Dados de Treino')
print('\nNúmero de Dados Treino')
print(len(treino_data))
print(treino_data['Label'].value_counts() / treino_data.shape[0] * 100)
print(treino_data.shape)

# teste_data
print('\nCarateristicas Dados de Teste')
print('\nNúmero de Dados de Teste')
print(len(teste_data))
print(teste_data['Label'].value_counts() / teste_data.shape[0] * 100)
print(teste_data.shape)

# TREINO

# Preparar Lista com todas as palavras do Dataset

vocabulary = list(set(treino_data['E-Mail'].sum()))
#print(len(vocabulary))


# Valores para a Fórmula de Naive Bayes

alpha = 1
Nvoc = len(treino_data.columns) - 3
probSpam = treino_data['Label'].value_counts()['spam'] / treino_data.shape[0]
probHam = treino_data['Label'].value_counts()['ham'] / treino_data.shape[0]
nSpam = treino_data.loc[treino_data['Label'] == 'spam', 'E-Mail'].apply(len).sum()
nHam = treino_data.loc[treino_data['Label'] == 'ham', 'E-Mail'].apply(len).sum()


def p_w_spam(word):
    if word in treino_data.columns:
        return (treino_data.loc[treino_data['Label'] == 'spam', word].sum() + alpha) / (nSpam + alpha * Nvoc)
    else:
        return 1


def p_w_ham(word):
    if word in treino_data.columns:
        return (treino_data.loc[treino_data['Label'] == 'ham', word].sum() + alpha) / (nHam + alpha * Nvoc)
    else:
        return 1


# Classificador
def classify(message):
    p_spam_given_message = probSpam
    p_ham_given_message = probHam
    for word in message:
        p_spam_given_message *= p_w_spam(word)
        p_ham_given_message *= p_w_ham(word)
    if p_ham_given_message > p_spam_given_message:
        return 'ham'
    elif p_ham_given_message < p_spam_given_message:
        return 'spam'
    else:
        return 'Indefinido'


# TESTE

teste_data['previsao'] = teste_data['E-Mail'].apply(classify)

correct = (teste_data['previsao'] == teste_data['Label']).sum() / teste_data.shape[0] * 100

toc = time.perf_counter()

# Dados de treino que foram classificados incorretamente
print('\nDados de treino que foram classificados incorretamente')
print(teste_data.loc[teste_data['previsao'] != teste_data['Label']])

print('\nPercentagem de identificações corretas Spam e Ham:')
print(correct)

print(f"\nTempo de cálculo total: {toc - tic:0.4f}s\n")