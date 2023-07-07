# Projeto final Processamento de Imagens
Projeto final da disciplina de processamento de imagens do PPGI da UTFPR-CP.
O projeto apresenta um código python com algumas interações que processam imagens de uma base de dados entre carros e motos e ao final do processamento exibe uma matriz de confusão e os dados relevantes como acurácia de acordo com os parametros escolhidos para a execução do programa.

## Equipe
Marcus Vinícius Santana Maziero - Aluno externo PPGI

## Dataset
O dataset escolhido foi: Car vs Bike Classification Dataset.
Um dos datasets recomentados pelo docente da disciplina.
Link do dataset - https://www.kaggle.com/datasets/utkarshsaxenadn/car-vs-bike-classification-dataset
## Descrição do projeto
O projeto realiza a classificação binária dos dados de motos (Bike) e carros (Car) de acordo com o normalizador escolhido e o classificador.

O descritor utilizado foi o SIFT(Scale Invariant Feature Transform), que é um método de extração dos recursos de uma imagem, onde os dados se tornam coordenadas com a localização desses recursos que não variam.

Nesse projeto foi utilizado do K-means que é um algoritmo de classificação (agrupamento) não supervisionado, ou seja não é necessário que os dados coletados sejam validados por algo ou alguém externo. Com isso é necessário definir o K - numero de clusters ou agrupamentos e o center de cada cluster.

*Atenção aqui foi utilizado um K - 100*

Para a execução do projeto pode-se escolher o classificador e o normalizador que será utilizado.

Os classificadores são:

* KNN (K-Nearest Neighbors) - procura o vizinho K mais proximo de um dado ponto no espaço do conjunto de dados.
* SVM (Maquina de Vetores Suporte - Support Vectors Machine) - e uma maquina de entrada e saida onde os dados que sao enviados como entrada sao mapeadas em um espaço multidimensional e encontra um hiperplano que separa os dados de entrada.

As opções de normalizadores para esse projeto foram:
* MinMaxScaler - consiste em transformar cada caracter´ıstica com valor mınimo em 0 e os valores maximos em 1, sendo que o restante e transformado em um valor decimal entre 0 e 1.
* StandardScaler -  são calculados a média e o desvio padrao da conjunto de amostras, em seguida é subtraída de cada amostra a media, o resultado então é divido pelo desvio padrão
* MaxAbsScaler - é uma técnica parecida com o Min-Max porem, somente os valores absolutos e positivos sao mapeados entre 0 e 1.
* RobustScaler - as estatísticas de mapeamento ocorrem pela escala de percentil, sendo assim aumenta a margem de onde estao os valores dos dados podendo ser ate negativo, exemplo: -3 a 2

LIMITAÇÕES

Atualmente se como valor de entrada para a escolha do normalizador e ou classificador for diferente das opções apresentadas a execução é finalizada.

FUTURO

Tratar os ocasionais erros e também melhor a qualidade do código como também a plotagem final dos dados.

## Repositório do projeto
https://github.com/MarcusVMaziero/image_processing
## Classificador e acurácia
SVM + MinMaxScaler:

## Instalação e Execução
Incluir os passos necessários para instalação de bibliotecas/dependências
necessárias para instalação e execução do projeto, bem como a estrutura de
pastas utilizadas para organização dos arquivos (código e dataset)
## Instruções de uso (opcional)
Incluir informações adicionais sobre o uso/execução do projeto