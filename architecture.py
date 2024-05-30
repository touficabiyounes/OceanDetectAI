# !/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pickle
import cv2
import shutil
import pickle
import pandas as pd
import PIL
import skimage
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
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
from sklearn.ensemble import StackingClassifier

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




# on a juste changer les parametres  et le resize pour avoir un resultat plus precis.


def raw_image_to_representation(image, representation):
    im = Image.open(image).convert('RGB')

    im = im.resize((400, 200))

    if representation == 'HC':
        r, g, b = im.split()
        hist_r = r.histogram()
        hist_g = g.histogram()
        hist_b = b.histogram()
        return hist_r, hist_g, hist_b

    elif representation == 'PX':
        transform = transforms.ToTensor()
        tensor = transform(im)
        return tensor

    elif representation == 'GC':
        gray_matrix = np.array(skimage.color.rgb2gray(im))
        return gray_matrix

    elif representation == 'HOG':
        gray_matrix = np.array(skimage.color.rgb2gray(im))
        hog_features, hog_image = hog(gray_matrix, orientations=8, pixels_per_cell=(12, 12),
                                      cells_per_block=(3, 3), visualize=True)

        return hog_features

    elif representation == 'HOG + HC':

        rgb_im = im.convert('RGB')
        r, g, b = rgb_im.split()
        hist_b = np.array(b.histogram()) /255

        gray_matrix = np.array(skimage.color.rgb2gray(im))

        hog_features, hog_image = hog(gray_matrix, orientations=9, pixels_per_cell=(8, 8),
                                      cells_per_block=(2, 2), visualize=True)
        combined_rep = np.concatenate((hist_b, hog_features))
        return combined_rep

    elif representation == 'PX + HC':
        rgb_im = im.convert('RGB')
        r, g, b = rgb_im.split()

        hist_r = np.array(r.histogram())
        hist_g = np.array(g.histogram())
        hist_b = np.array(b.histogram())

        transform = transforms.ToTensor()
        tensor = transform(im)

        combined_rep = np.concatenate((np.array(tensor[0]).flatten(), hist_r, hist_b, hist_g))

        return combined_rep

    else:
        raise ValueError("Invalid representation: {}".format(representation))



# l'augmentation des data etaient inutile en fin de compte.

def augment_image_pil(image_path):

    image = Image.open(image_path).convert('RGB')
    brightness_enhancer = ImageEnhance.Brightness(image)
    brighter_image = brightness_enhancer.enhance(2)

    contrast_enhancer = ImageEnhance.Contrast(image)
    higher_contrast_image = contrast_enhancer.enhance(2)

    color_enhancer = ImageEnhance.Color(image)
    higher_color_image = color_enhancer.enhance(2)

    im_gaussian = image.filter(ImageFilter.GaussianBlur(3))

    return [image, brighter_image, higher_contrast_image, higher_color_image, im_gaussian]


"""
Returns a relevant structure embeddibng train images described according to the
specified representation and associate each image (name or/and location) to its label.
-> Representation can be (to extend)
'HC': color histogram
'PX': tensor of pixels
'GC': matrix of gray pixels
other to be defined
--
input = where are the examples, which representation of the data must be produced ?
output = a relevant structure (to be discussed, see below) where all the images of the
directory have been transformed, named and labelled according to the directory they are
stored in (the structure lists all the images, each image is made up of 3 informations,
namely its name, its representation and its label)
This structure will later be used to learn a model (function learn_model_from_dataset)
-- uses function raw_image_to_representation
"""


def load_transform_label_train_dataset(directory, representation):
    dataset = []

    for repository in os.listdir(directory):
        if repository == "Ailleurs":
            label = 0
        else:
            label = 1
        for image_name in os.listdir(os.path.join(directory, repository)):
            image_path = os.path.join(directory, repository, image_name)
            image_representation = raw_image_to_representation(image_path, representation)
            if not isinstance(image_representation, np.ndarray):
                image_representation = np.asarray(image_representation)
            image_representation_flat = image_representation.flatten()
            dataset.append({'image_name': image_name, 'representation': image_representation_flat, 'label': label})

    return dataset




"""
Returns a relevant structure embedding test images described according to the
specified representation.
-> Representation can be (to extend)
'HC': color histogram
'PX': tensor of pixels
'GC': matrix of gray pixels
other to be defined
--
input = where are the data, which represenation of the data must be produced ?
output = a relevant structure, preferably the same chosen for function load_transform_label_train_data
-- uses function raw_image_to_representation
-- must be consistant with function load_transform_label_train_dataset
-- while be used later in the project
"""


def load_transform_test_data(directory, representation):
    test_data = []

    for image_name in os.listdir(directory):
        image_path = os.path.join(directory, image_name)
        image_representation = raw_image_to_representation(image_path, representation)
        if not isinstance(image_representation, np.ndarray):
            image_representation = np.asarray(image_representation)

        image_representation_flat = image_representation.flatten()
        image_info = {'image_name': image_name, 'representation': image_representation_flat}
        test_data.append(image_info)
    return test_data


"""
Learn a model (function) from a pre-computed representation of the dataset, using the algorithm
and its hyper-parameters described in algo_dico
For example, algo_dico could be { algo: 'decision tree', max_depth: 5, min_samples_split: 3 }
or { algo: 'multinomial naive bayes', force_alpha: True }
--
input = transformed labelled dataset, the used learning algo and its hyper-parameters (better a dico)
output =  a model fit with data
"""


def learn_model_from_dataset(train_dataset, algo_dico):
    labels = [item['label'] for item in train_dataset]

    data = [item['representation'] for item in train_dataset]
    # x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

    algo = algo_dico.get('algo')

    if algo == 'decision tree':
        param_grid = {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        model = DecisionTreeClassifier()
        model = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
    elif algo == 'random forest':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        model = RandomForestClassifier()
        model = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')

    elif algo == 'multinomial naive bayes':
        force_alpha = algo_dico.get('force_alpha', False)
        model = MultinomialNB()
        param_grid = {
            'alpha': [0.1, 0.5, 1.0, 2.0],
            'fit_prior': [True, False]
        }
        model = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')

    elif algo == 'SVC':
        model = SVC()
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto'],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
        }
        model = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')

    elif algo == 'VotingClassifier':
        multi = MultinomialNB(alpha=algo_dico['alpha'])
        decision_tree_clf = DecisionTreeClassifier(
            criterion=algo_dico['criterion_dt'],
            max_depth=algo_dico['max_depth_dt'],
            random_state=algo_dico['random_state_dt']
        )
        rnd_clf = RandomForestClassifier(
            bootstrap=algo_dico['bootstrap_rf'],
            n_estimators=algo_dico['n_estimators_rf'],
            criterion=algo_dico['criterion_rf'],
            max_features=algo_dico['max_features_rf'],
            min_samples_leaf=algo_dico['min_samples_leaf_rf'],
            max_depth=algo_dico['max_depth_rf'],
            min_samples_split=algo_dico['min_samples_split_rf'],
            random_state=algo_dico['random_state_rf'],
            n_jobs=algo_dico['n_jobs_rf']
        )
        poly_svm_chi2_clf = SVC(kernel=algo_dico['kernel_svm'])
        modelG = xgb.XGBClassifier()
        modelA = AdaBoostClassifier()
        model = VotingClassifier(
            estimators=[
                ('decision_tree_clf', decision_tree_clf),
                ('poly', poly_svm_chi2_clf),
                ('rnd_clf', rnd_clf),
                ('MNB', multi),
                ('modelG', modelG),
                ('modelA', modelA)
            ],
            voting="hard"
        )
    elif algo == 'knn':
        model = KNeighborsClassifier(n_neighbors=5)
        param_grid = {
            'n_neighbors': [3, 5, 7, 9],  # Number of neighbors
            'weights': ['uniform', 'distance'],  # Weight function used in prediction
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
        }  # Algorithm used to compute the nearest neighbors
        model = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')


    elif algo == 'xgb':
        param_grid = {
            'learning_rate': [0.1, 0.01, 0.001],
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
        }

        model = xgb.XGBClassifier()
        # model = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')

    elif algo == 'adab':
        model = AdaBoostClassifier()
    elif algo == 'stacking':
        multi = MultinomialNB(alpha=algo_dico['alpha'])
        decision_tree_clf = DecisionTreeClassifier(
            criterion=algo_dico['criterion_dt'],
            max_depth=algo_dico['max_depth_dt'],
            random_state=algo_dico['random_state_dt']
        )
        rnd_clf = RandomForestClassifier(
            bootstrap=algo_dico['bootstrap_rf'],
            n_estimators=algo_dico['n_estimators_rf'],
            criterion=algo_dico['criterion_rf'],
            max_features=algo_dico['max_features_rf'],
            min_samples_leaf=algo_dico['min_samples_leaf_rf'],
            max_depth=algo_dico['max_depth_rf'],
            min_samples_split=algo_dico['min_samples_split_rf'],
            random_state=algo_dico['random_state_rf'],
            n_jobs=algo_dico['n_jobs_rf']
        )
        poly_svm_chi2_clf = SVC(kernel=algo_dico['kernel_svm'])
        modelG = xgb.XGBClassifier()
        modelA = AdaBoostClassifier()
        base_classifiers = [
            ('decision_tree_clf', decision_tree_clf),
            ('poly', poly_svm_chi2_clf),
            ('rnd_clf', rnd_clf),
            ('MNB', multi),
            ('modelG', modelG),
            ('modelA', modelA)]
        stacking_classifier = StackingClassifier(estimators=base_classifiers, final_estimator=modelG)
        model = stacking_classifier

    model.fit(data, labels)

    # pickle.dump(model, open('model.pkl', 'wb'))

    # prediction = model.predict(testdata)

    # testscor = accuracy_score(prediction, testlabels)
    # print('test_score', testscor)
    return model


"""
Given one example (previously loaded with its name and representation),
computes its class according to a previously learned model.
--
input = representation of one data, the learned model
output = the label of that one data (+1 or -1)
-- uses the model learned by function learn_model_from_dataset
"""


def predict_example_label(example, model):
    label = model.predict([example])[0]
    return label


"""
Computes a structure that computes and stores the label of each example of the dataset,
using a previously learned model.
--
input = a structure embedding all transformed data to a representation, and a model
output =  a structure that associates a label to each identified data (image) of the input dataset
"""


def predict_sample_label(dataset, model):
    predictions = {}
    for entry in dataset:
        if 'image_name' in entry and 'representation' in entry:
            representation = entry['representation']
            image_name = entry['image_name']

            predictions[image_name] = predict_example_label(representation, model)

    return predictions


"""
Estimates the accuracy of a previously learned model using train data,
either through CV or mean hold-out, with k folds.
input = the train labelled data as previously structured, the type of model to be learned
(as in function learn_model_from_data), and the number of split to be used either
in a hold-out or by cross-validation
output =  The score of success (betwwen 0 and 1, the higher the better, scores under 0.5
are worst than random guess)
"""


def estimate_model_score(train_dataset, algo_dico, k):
    x_train = [entry['representation'] for entry in train_dataset]
    y_train = [entry['label'] for entry in train_dataset]

    algo = algo_dico.get('algo')

    if algo == 'decision tree':
        max_depth = algo_dico.get('max_depth', None)
        min_samples_split = algo_dico.get('min_samples_split', 2)
        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
    elif algo == 'multinomial naive bayes':
        force_alpha = algo_dico.get('force_alpha', False)
        model = MultinomialNB(alpha=1.0 if force_alpha else 0.0)

    elif algo == 'xgb':
        model = xgb.XGBClassifier()
    else:
        raise ValueError("Unsupported algorithm")

    cv = KFold(n_splits=k, shuffle=True, random_state=42)

    scores = cross_val_score(model, x_train, y_train, cv=cv)
    score_mean = scores.mean()
    print(score_mean)
    return score_mean


def save_model_to_file(model, directory):
    filename = 'modelM3.pkl'
    filepath = os.path.join(directory, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)

    print(f"Model saved to: {filepath}")


def load_model_from_file(filepath):

    with open(filepath, 'rb') as f:
        model = pickle.load(f)

    return model


def predict(model, x):
    pre = model.predict(x)
    return pre

def write_predictions(directory, filename, predictions, algo_dico):
    try:
        filepath = os.path.join(directory, filename)
        print("File path:", filepath)

        with open(filepath, 'w') as file:
            file.write(f"Learning Algorithm: {algo_dico['algo']}\n")
            for key, value in algo_dico.items():
                if key != 'algo':
                    file.write(f"{key}: {value}\n")
            for dataName, label in predictions.items():
                if label == 0:
                    label = -1
                file.write(dataName + ' ' + str(label) + '\n')
        return "OK"
    except IOError as e:

        print("Error writing to the file:", str(e))
        return "not OK"

    """
    wright the prediction of test data in txt file
    """


def trained_model_to_predict(modelpath, derectry, filename, algo_dico, testdatapath):
    model = load_model_from_file(modelpath)
    testdata = load_transform_test_data(testdatapath, 'HOG + HC')
    predictions = predict_sample_label(testdata, model)
    write_predictions(derectry, filename, predictions, algo_dico)


