import architecture

import pickle
import cv2

import PIL
import skimage
from PIL import Image
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, KFold, train_test_split, GridSearchCV

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from skimage.feature import hog

from torchvision import transforms

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
from sklearn.preprocessing import RobustScaler


'''
La première façon de lancer le code est d'utiliser le modèle déjà sauvegardé dans le fichier Model3.
 La deuxième façon consiste à décommenter les autres lignes pour entraîner le modèle et 
 commenter la ligne pour lancer le code de l'étape 1
 '''

if __name__ == '__main__':

    model = {'algo': 'xgb'}
    #architecture.trained_model_to_predict('modelM3.pkl', '', 'METecs.txt', model, 'RawDataCC3')

    trained_data = architecture.load_transform_label_train_dataset('Data', 'HOG + HC')
    print(trained_data)
    learn_model = architecture.learn_model_from_dataset(trained_data, model)

    architecture.save_model_to_file(learn_model,'')
    data_to_test = architecture.load_transform_test_data('RawDataCC3', 'HOG + HC')

    predictions = architecture.predict_sample_label(data_to_test, learn_model)
    architecture.write_predictions('', 'METecs.txt', predictions, model)


