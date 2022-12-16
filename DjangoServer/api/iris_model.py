import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder

'''
Iris Species
Classify iris plants into three species in this classic dataset
'''


class IrisModel(object):
    def __init__(self):
        global iris, _X, _Y
        iris = datasets.load_iris()
        print(f'type {type(iris)}')  # type <class 'sklearn.utils._bunch.Bunch'>
        _X = iris.data
        _Y = iris.target

    def hook(self):
        self.spec()
        self.create_model()

    def spec(self):
        print(" --- 1.Features ---")
        print(iris['feature_names'])
        print(" --- 2.target ---")
        print(iris['target'])
        print(" --- 3.print ---")
        print(iris)
        '''
        Shape (150, 6)
        ['Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm','Species']

        print(" --- 1.Shape ---")
        print(iris.shape)
        print(" --- 2.Features ---")
        print(iris.columns)
        print(" --- 3.Info ---")
        print(iris.info())
        print(" --- 4.Case Top1 ---")
        print(iris.head(1))
        print(" --- 5.Case Bottom1 ---")
        print(iris.tail(3))
        print(" --- 6.Describe ---")
        print(iris.describe())
        print(" --- 7.Describe All ---")
        print(iris.describe(include='all'))
        '''

    def create_model(self):
        X = _X
        Y = _Y
        enc = OneHotEncoder()
        Y_1hot = enc.fit_transform(Y.reshape(-1, 1)).toarray()
        model = Sequential()
        model.add(Dense(4, input_dim=4, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X, Y_1hot, epochs=300, batch_size=10)
        print('Model Training is completed')

        file_name = 'save/iris_model.h5'
        model.save(file_name)
        print(f'Model Saved in {file_name}')


iris_menu = ["Exit",  # 0
             "hook"]  # 1
iris_lambda = {
    "1": lambda x: x.hook(),
}
if __name__ == '__main__':
    iris_model = IrisModel()
    while True:
        [print(f"{i}. {j}") for i, j in enumerate(iris_menu)]
        menu = input('메뉴선택: ')
        if menu == '0':
            print("종료")
            break
        else:
            try:
                iris_lambda[menu](iris_model)
            except KeyError as e:
                if 'some error message' in str(e):
                    print('Caught error message')
                else:
                    print("Didn't catch error message")