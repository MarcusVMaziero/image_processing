# Projeto final Processamento de Imagens
Projeto final da disciplina de processamento de imagens do PPGI da UTFPR-CP.

Link com o Vídeo de apresentação: https://drive.google.com/file/d/1TdA0yE-_XlV-p2A4qWMi0YK8LD3jBL09/view?usp=sharing

O projeto apresenta um código python com algumas interações que processam imagens de uma base de dados entre carros e motos e ao final do processamento exibe uma matriz de confusão e os dados relevantes como acurácia de acordo com os parametros escolhidos para a execução do programa.

## Equipe
Marcus Vinícius Santana Maziero - Aluno externo PPGI

## Dataset
O dataset escolhido foi: Car vs Bike Classification Dataset.
Um dos datasets recomentados pelo docente da disciplina.
Link do dataset - https://www.kaggle.com/datasets/utkarshsaxenadn/car-vs-bike-classification-dataset
O dataset possui no total 4.000 imagens, sendo 50% de Car e 50% de Bike.
Para a execução do projeto foi divido em 2 pastas a training(80% da base) e test(20% da base). A divisão foi aleatória porém garantindo que em cada pasta estaria os elementos correspondentes.
A decisão de realizar a divisão foi para facilitar o entendimento do discente no inicio do projeto.
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
Com classificador SVM

SVM + MinMaxScaler:

 <img src="https://github.com/MarcusVMaziero/image_processing/blob/main/accuracy/svm%2Bminmax-ac.png">

SVM + StandardScaler:

 <img src="https://github.com/MarcusVMaziero/image_processing/blob/main/accuracy/svm%2Bstand.png">

SVM + MaxAbsScaler:

 <img src="https://github.com/MarcusVMaziero/image_processing/blob/main/accuracy/svm%2Bmaxabs.png">

SVM + RobustScaler:

 <img src="https://github.com/MarcusVMaziero/image_processing/blob/main/accuracy/svm%2Breso.png">

Com classificador KNN

KNN + MinMaxScaler:

 <img src="https://github.com/MarcusVMaziero/image_processing/blob/main/accuracy/knn%2Bminmax.png">

KNN + StandardScaler:

 <img src="https://github.com/MarcusVMaziero/image_processing/blob/main/accuracy/knn%2Bstand.png">

KNN + MaxAbsScaler:

 <img src="https://github.com/MarcusVMaziero/image_processing/blob/main/accuracy/knn%2Bmaxabs.png">

KNN + RobustScaler:

 <img src="https://github.com/MarcusVMaziero/image_processing/blob/main/accuracy/knn%2Breso.png">

Resultado

Os melhores resultados foram:

Utilizando o classificador SVM e o normalizador StandardScaler a acaurácia foi de 0.79

Utilizando o classificador KNN e o normalizador RobustScaler a acaurácia foi de 0.79

Os piores resultados foram:

Utilizando o classificador SVM e o normalizador MinMax a acaurácia foi de 0.50

Utilizando o classificador SVM e o normalizador MaxAbs a acaurácia foi de 0.50

Para visualizar a matriz de confusão acesse: https://github.com/MarcusVMaziero/image_processing/tree/main/accuracy/confu

## Instalação e Execução
Incluir os passos necessários para instalação de bibliotecas/dependências
necessárias para instalação e execução do projeto, bem como a estrutura de
pastas utilizadas para organização dos arquivos (código e dataset)
Clonar o projeto do repo: https://github.com/MarcusVMaziero/image_processing
* Para executar o projeto é necessário possuir o Python em sua máquina sendo a versão minima 3.8.10
* O projeto pode ser aberto em qualquer editor de texto ou IDE - ex: VSCode / PyCharm...
* Após é necessário importar os pacotes externos que são utilizados, para isso utilize o pip ou outro gerenciador.
* Com isso basta executar o projeto (arquivo: classification.py) selecionar as opções de normalizador e classificador.

Estrutura do projeto:

    -classification.py
    -dataset
        -test
            -Bike
            -Car
        -training
            -Bike
            -Car

Vale lembrar que o dataset já esta divido com a regra 80/20, sendo 80% para treinamento e 20% para teste

ATENÇÃO - NÃO ALTERE NENHUMA ORDEM OU COMPOSIÇÃO DA PASTA dataset (POIS IRÁ OCASIONAR UM PROBLEMA E/OU RESULTADOS INCORRETOS)

É legal executar mais de uma vez e analisar as diferenças entre os normalizadores e classificadores e como isso afeta a acurácia.
## Instruções de uso (opcional)
O código desse projeto pode servir para outras comparações binárias de outros datasets, basta realizar algumas alterações.
