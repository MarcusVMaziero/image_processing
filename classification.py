import os
import cv2
import numpy as np
from scipy.spatial import distance
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix
from progress.bar import Bar
import warnings
warnings.filterwarnings("ignore")

trainPath = 'images/cats_dogs_light/train/'
testPath = 'images/cats_dogs_light/test/'

def getData(path):
    images = []
    labels = []
    if os.path.exists(path):
        fileList = os.listdir(path)
        for fileName in fileList:
            label = os.path.basename(fileName).split('.')[0]
            if (label == 'cat'):
                labels.append(0)
            else:
                labels.append(1)
            colorImage = cv2.imread(path+fileName)
            grayImage = cv2.cvtColor(colorImage, cv2.COLOR_BGR2GRAY) 
            blurImage = cv2.medianBlur(grayImage,3)
            gray = np.array(blurImage,dtype='float32')/255 #normalized [0,1]
            images.append(blurImage)
    return images, labels

print('[INFO] Getting train data...')
trainImages, trainLabels = getData(trainPath)
trainImages = np.array(trainImages)
#print(f'Shape of images = {trainImages.shape}')

def getDescriptors(images):
    sift = cv2.SIFT_create()
    descriptorListPerImage = []
    descriptorListOfAllImages = []
    with Bar('[INFO] Computing descriptors...', max = len(images)) as bar:
        for image in images:
            keypoints, descriptors = sift.detectAndCompute(image,None)
            #print(f'Descriptors length = {len(descriptors[0])}')
            descriptorListPerImage.append(descriptors)
            descriptorListOfAllImages.extend(descriptors)
            bar.next()
    return descriptorListPerImage, descriptorListOfAllImages

print('[INFO] Getting descriptors from train images...')
descriptorListPerTrainImage, descriptorListOfAllTrainImages = getDescriptors(trainImages)

descriptorListPerTrainImage = np.array(descriptorListPerTrainImage)
print(f'Shape of descriptorListPerTrainImage = {descriptorListPerTrainImage.shape}')

descriptorListOfAllTrainImages = np.array(descriptorListOfAllTrainImages)
print(f'Shape of descriptorListOfAllTrainImages = {descriptorListOfAllTrainImages.shape}')


print('[INFO] Clustering the descriptorList for train images...')
k = 200 
kmeans = KMeans(n_clusters = k, n_init='auto')
kmeans.fit(descriptorListOfAllTrainImages)
centers = kmeans.cluster_centers_ 
#print(f'Size of each center={len(centers[0])}')
print('[INFO] Clustering done!')

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
    for i in range(len(descriptorList)):
        histogram = np.zeros(len(centers))
        for d in range(len(descriptorList[i])):
            #print(f'Shape of descriptorList[{i}] = {descriptorList[i].shape}')
            #print(f'Tamanho descritor {len(descriptorList[i][d])}')
            descriptor = descriptorList[i][d]
            idx = find_index(descriptor,centers)
            histogram[idx] += 1
        histogramList.append(histogram)
    return histogramList

print('[INFO] Getting histogram list from train images...')
trainData = getHistogramList(descriptorListPerTrainImage,centers)

print('[INFO] Training the SVM model...')
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(trainData, trainLabels)

print('[INFO] Getting test data...')
testImages, testLabels = getData(testPath)
testImages = np.array(testImages)

print('[INFO] Getting descriptors from test images...')
descriptorListPerTestImage, descriptorListOfAllTestImages = getDescriptors(testImages)
descriptorListPerTestImage = np.array(descriptorListPerTestImage)
print(f'Shape of descriptorListPerTestImage = {descriptorListPerTestImage.shape}')

print('[INFO] Getting histogram list from test images...')
testData = getHistogramList(descriptorListPerTestImage,centers)

print('[INFO] Predicting...')
classification = clf.predict(testData)

print(classification)

print(confusion_matrix(testLabels,classification))
print(classification_report(testLabels,classification))

