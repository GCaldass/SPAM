import string
import time

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords

########################################################################################################################
# Ler o ficheiro csv que contem os emails e mudar o nome dos atributos

# VER ENCODING - latin-1 ou UTF-8
data_emails = pd.read_csv("spam.csv", encoding='cp1252')
data_emails = data_emails.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
data_emails = data_emails.rename(columns={'v1': 'type', 'v2': 'email'})
data_emails.head()

train = data_emails
train.head(4)
print(data_emails)
########################################################################################################################

tic = time.perf_counter()

mess = 'sample message!...'
nopunc = [char for char in mess if
          char not in string.punctuation]  # retornar caracter a caracter da var "mess" para a variavel "nopunc" caso o caracter não seja um ponto
nopunc = ''.join(nopunc)

# importar a library stopwords para mais a frente as removermos dos nossos dados

nltk.download('stopwords')
stopwords.words('english')


# função para remover a pontuação e as palavras que não nos interessam
def text_process(mess):
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


data_emails['email'].head(5).apply(text_process)  # aplicação da funcão text_process no dataset

# listas
bag_of_words = []
bag_of_words_final = []

# para cada frase lida do dataset emails, vamos guarda-la no bag of words palavra a palavra
for i in data_emails['email']:
    i.rstrip()  # strip white space
    words = i.split()  # split lines of words and make list
    bag_of_words.extend(words)  # make the list from 4 lists to 1 list

bag_of_words.sort()  # sort list

for word in bag_of_words:  # for each word in line.split
    if word not in bag_of_words_final:  # if a word isn't in line.split
        bag_of_words_final.append(word)  # juntar a palavra à nossa lista de palavras final

# remover caracteres que não interessam no saco de palavras
remover_caracteres = str.maketrans('', '',
                                   '@#%!*;,.-:_$/|?\'&''()')  # recebe dois parametros vazios, e no terceiro parametro contém os caracteres que queremos remover
bag_of_words_final = [caracter.translate(remover_caracteres) for caracter in bag_of_words_final]
bag_of_words_final = [palavras_em_branco for palavras_em_branco in bag_of_words_final if
                      palavras_em_branco]  # para remover os espaços em branco

# remover os numeros do bag_of_words_final
bag_of_words_final = [palavras for palavras in bag_of_words_final if not any(caracter.isdigit() for caracter in
                                                                             palavras)]  # para cada palavra, verifica se algum dos caracteres é um nº, e se for retira o nº

# remover palavras que não nos interessam

for word in bag_of_words_final:
    if word == 'XXX' or word == 'gt' or word == 'lt' or word == 'ltDECIMALgt' or word == 'ltEMAILgt' or word == 'ltTIMEgt' or word == 'ltURLgt':
        bag_of_words_final.remove(word)

print("Saco de palavras:", bag_of_words_final)

########################################################################################################################
# Separação dos dados em treino e teste
sms_dataset = open("spam.csv", "r", encoding="latin-1")

emails = [item.strip('\n') for item in sms_dataset]

print(emails)  # visualizar o conteudo de emails
# verifica-se que o nome das colunas do dataset também estão inseridas nos emails...
emails = emails[1:]  # "remover" o nome das colunas

# Proceder à separação
###################### DADOS DE TREINO
train_data = []  # lista que vai guardar os dados de treino

train_perc = 80
train_len = int((train_perc * len(emails)) / 100)

print('Comprimento dos dados de treino', train_len)

for i in range(train_len):  # guardar 80 % dos dados nos dados de treino
    train_data.append(emails[i])

for email in train_data:  # remover dos emails cada email que percente aos dados de treino
    emails.remove(email)

###################### DADOS DE TESTE
test_data = []  # lista que vai guardar os dados de teste
test_len = len(emails)
print('Comprimento dos dados de validação', test_len)

# guardar os dados que sobram na lista que recebe os dados de teste
for i in range(test_len):
    test_data.append(emails[i])

for email in test_data:
    emails.remove(email)  # remover dos emails cada email que percente aos dados de teste

########################################################################################################################
w = [0] * len(bag_of_words_final)  # pesos..

b = 0  # bias
i = 1
i2 = 0
a = 0
ListaX = []  # Lista de vetores com freq. abs. - Treino
ListaFrequencias = [0] * len(
    bag_of_words_final)  # [0,0,2,0,0,0,5,...] - lista (vetor) com nº de vezes que cada palavra aparece em Train[label, frase]

# Funcao para Calcular as frequencias absolutas

strSplit = []


def CalcularFreqAbs(fraseX, Lista):
    i = 0
    strSplit = fraseX.split()
    for palavra in bag_of_words_final:
        ListaFrequencias[i] = strSplit.count(palavra)
        i += 1
    Lista.append(ListaFrequencias)  # ListaX.append(Vfreq)

    # Calcular Frequencia Absoluta


for item in train:
    CalcularFreqAbs(item[1], ListaX)

#  arrY=[0]*len(train)
list_type = []  # lista que vai guardar as labels (y)

for t in train['type']:
    if t == "spam":
        list_type.append(-1)
    else:
        list_type.append(1)

print('Lista de labels(contem as representaçoes das labels de spam e ham que sao -1 e 1 respetivamente):\n', list_type)

print('Número de Mensagens Classificadas com SPAM - ' + str(list_type.count(-1)))
print('Número de Mensagens Classificadas com HAM - ' + str(list_type.count(1)))
########################################################################################################################
# ALGORITMO DO PERCEPTRÃO
i = 0
d = 0  # indice do w e x
MaxIter = 3
for i in range(MaxIter):
    iV = 0  # (vetor)
    for x in ListaX:  # ciclo mesma length que Treino - x= vetor com nº de vezes que cada palavra aparece em Train[label, frase]
        for d in range(len(bag_of_words_final)):
            a += w[d] * x[d]
        a = a + b
        if list_type[iV] * a <= 0:  # caso o y(ListaLabels) seja menor ou igual a zero, procedemos à atualização
            for d2 in range(len(bag_of_words_final)):
                w[d2] = w[d2] + list_type[iV] * x[d2]  # atualizar pesos
                b = b + list_type[iV]  # atualizar bias
        iV += 1
print('Pesos: ', w)
print('Bias: ', b)

########################################################################################################################
# TESTAR O PERCEPTRÃO
list_freqAbsolutas_test = []
a2 = 0

# calcular a frequencia absoluta dos dados de test
for item in test_data:
    CalcularFreqAbs(item[1], list_freqAbsolutas_test)

    # Algoritmo Perceptron Test - with test
for x1 in list_freqAbsolutas_test:
    for d in range(len(list_freqAbsolutas_test)):  # ciclo Somatório
        a2 += w[d] * x1[d] + b

a2 = np.sign(a2)  # sign function returns (-1 if x < 0, 0 if x==0, 1 if x > 0)
print(a2)

########################################################################################################################
# RESULTADO FINAL DO ALGORITMO DO PERCEPTRÃO
iv = 0

for x1 in list_freqAbsolutas_test:
    for d in range(len(list_freqAbsolutas_test)):
        a2 += w[d] * x1[d] + b
    a2 = a2 + b
    a2 = np.sign(a2)
    # print(a2)
    # print("y: ", x1[iv])
    # Contar nº de vezes acerta (precisão)
    iv += 1
    # print(iv)

print("Precisão: " + str(iv))

toc = time.perf_counter()

##Métricas
print(f"\nTempo de cálculo total: {toc - tic:0.4f}s\n")
# Alex - 13.2944s
