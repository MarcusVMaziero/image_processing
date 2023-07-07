import cv2
import numpy as np
from scipy.spatial import distance
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from progress.bar import Bar
import glob
import warnings
warnings.filterwarnings("ignore")

trainPath = 'dataset/training/'
testPath = 'dataset/test/'

typeCar = 'Car'
typeBike = 'Bike'

def getImages(file):
    colorImage = cv2.imread(file)
    grayImage = cv2.cvtColor(colorImage, cv2.COLOR_BGR2GRAY) 
    blurImage = cv2.medianBlur(grayImage,3)
    return blurImage

#Como esta dividido em 2 pastas diferentes não é validado o label, sendo BIKE 0 e CAR 1 - até porque temos algumas imagens que não estão com o fileName correto mas estão na pasta certa
def readData(path):
    patterns = ["*.*"]
    images = []
    labels=[]
    for pattern in patterns:
        for file in glob.glob(path+typeBike+'/' + pattern):
            images.append(getImages(file))
            labels.append(0)
    for pattern in patterns:
        for file in glob.glob(path+typeCar+'/' + pattern):
            images.append(getImages(file))
            labels.append(1)
    print('[INFO] Quantidade de itens:', len(images))
    return images, labels

def getDescriptors(images):
    sift = cv2.SIFT_create()
    siftDescriptorsList = []
    bar = Bar('[INFO] Extraindo SIFT descritor...',max=len(images),suffix='%(index)d/%(max)d  Duration:%(elapsed)ds')
    for image in images:
        #NÃO ESQUECER aqui se a imagem não tem 8bits vai dar erro
        keypoints, descriptors = sift.detectAndCompute(image,None)
        siftDescriptorsList.append(descriptors)
        bar.next()
    bar.finish()
    return np.array(siftDescriptorsList,dtype=object)

def find_index(descriptor, center):
    dist1 = 0
    dist2 = 0
    for i in range(len(center)):
        if(i == 0):
           dist1 = distance.euclidean(descriptor, center[i]) 
        else:
            dist = distance.euclidean(descriptor, center[i]) 
            if(dist < dist1):
                dist2 = i
                dist1 = dist
    return dist2

def getHistogramList(descriptorList, centers):
    histogramList = []
    print('[INFO] Extraindo SIFT descritor...')
    for i in range(len(descriptorList)):
        histogram = np.zeros(len(centers))
        for d in range(len(descriptorList[i])):
            descriptor = descriptorList[i][d]
            idx = find_index(descriptor,centers)
            histogram[idx] += 1
        histogramList.append(histogram)
    print('[INFO] Finalizou a extração SIFT descritor...')
    return histogramList



def training(scaler, classifier):
    print('[INFO] Iniciando leitura para treinamento...')
    trainImages, trainLabels = readData(trainPath)

    trainImages = np.array(trainImages, dtype=object)
    print('[INFO] Capturando os descritores para treinamento...')
    trainSiftDescriptors = getDescriptors(trainImages)

    print('[INFO] Carregando a lista de descritores para treinamento...')
    k = 10 
    kmeans = KMeans(n_clusters = k, n_init='auto')
    kmeans.fit(np.vstack(trainSiftDescriptors))
    centers = kmeans.cluster_centers_ 
    print('[INFO] Carregamento da lista de descritores realizada com sucesso!')

    print('[INFO] Capturando o historiograma para a base de treinamento...')
    trainData = getHistogramList(trainSiftDescriptors,centers)

    print('[INFO] Treinando no modelo Escolhido...')
    clf = make_pipeline(scaler, classifier)
    clf.fit(trainData, trainLabels)
    return clf, centers

def test(clf, centers):
    print('[INFO] Iniciando leitura para teste...')
    testImages, testLabels = readData(testPath)
    testImages = np.array(testImages, dtype=object)

    print('[INFO] Capturando os descritores para teste...')
    descriptorListPerTestImage = getDescriptors(testImages)
    descriptorListPerTestImage = np.array(descriptorListPerTestImage)

    print('[INFO] Capturando o historiograma para a base de teste...')
    testData = getHistogramList(descriptorListPerTestImage,centers)

    print('[INFO] Previsão...')
    classification = clf.predict(testData)

    print(classification)

    conf_matrix = confusion_matrix(testLabels,classification)

    class_names = [ 'Bike','Car']
 
    plot_confusion_matrix(conf_mat=conf_matrix, figsize=(6, 6), cmap=plt.cm.Greens)
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.xticks(np.arange(2), class_names)
    plt.yticks(np.arange(2), class_names)
    plt.show()
    print(conf_matrix)
    print(classification_report(testLabels,classification))

#EXECUÇÃO MAIN

print ("Vamos iniciar o processamento de imagens da base BIKE and CAR um dataset com um total de 4.000 imagens, sendo metade de carros e metade de motos \n A base esta divida em 80% treinamento e 20% teste (3.600 - treinamento / 400 teste)")
print ("Digite qual opção de normalização você deseja aplicar para o conjunto de treinamento: \n 1 = MinMaxScaler \n 2 = StandardScaler \n 3 = MaxAbsScaler \n 4 = RobustScaler")

# Escolha umas das 4 técnicas de normalização existentes
# 1 = MinMaxScaler, 2 = StandardScaler, 3 = MaxAbsScaler, 4 = RobustScaler
selectedNormalization = int(input())

if selectedNormalization == 1:
    scaler = preprocessing.MinMaxScaler()
if selectedNormalization == 2:
    scaler = preprocessing.StandardScaler()
if selectedNormalization == 3:
    scaler = preprocessing.MaxAbsScaler()
if selectedNormalization == 4:
    scaler = preprocessing.RobustScaler()

print("Certo agora vamos escolher o Classificador: \n 1 = SVM \n 2 = KNN")
selectedClassifier = int(input())

if selectedClassifier == 1:
    classifier = SVC(gamma='auto')
if selectedClassifier == 2:
    classifier = KNeighborsClassifier(n_neighbors = 3)


clf, centers = training(scaler, classifier)
test(clf, centers)

    
    