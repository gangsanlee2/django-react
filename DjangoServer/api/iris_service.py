import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
from keras.saving.save import load_model
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
import os


class IrisService(object):
    def __init__(self):
        global model, graph, target_names
        model = load_model(os.path.join(os.path.abspath("save"), "iris_model.h5"))
        target_names = datasets.load_iris().target_names

    def service_model(self, features): # features = []
        features = np.reshape(features, (1, 4))
        Y_prob = model.predict(features, verbose=0)
        print(f'type is {type(Y_prob)}')
        predicted = Y_prob.argmax(axis=-1)
        return predicted[0]  # p-value 가 가장 높은 것

iris_menu = ["Exit", #0
                "hook"] #1
iris_lambda = {
    "1" : lambda x: x.hook(),
}
if __name__ == '__main__':
    iris = IrisService()
    while True:
        [print(f"{i}. {j}") for i, j in enumerate(iris_menu)]
        menu = input('메뉴선택: ')
        if menu == '0':
            print("종료")
            break
        else:
            try:
                iris_lambda[menu](iris)
            except KeyError as e:
                if 'some error message' in str(e):
                    print('Caught error message')
                else:
                    print("Didn't catch error message")