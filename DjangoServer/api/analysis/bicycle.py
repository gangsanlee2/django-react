import pandas as pd
from api.analysis.common import Common
from api.analysis.dataset import Dataset


class BicycleModel(object):

    dataset = Dataset()

    def __init__(self):
        pass

    def __str__(self):
        return f""

    def preprocess(self):
        pass

    def new_model(self, fname) -> object:
        this = self.dataset
        this.context = './data/bicycle/'
        this.fname = fname
        df = pd.read_csv(this.context + this.fname)
        print(f'데이터프레임 내부 보기 : \n{df}')
        return df

    def create_train(self):
        pass

    def create_label(self):
        pass

class Tempalate(object):

    def __init__(self):
        pass

    def __str__(self):
        return f""

class BicycleController(object):

    model = BicycleModel()

    def __init__(self):
        pass

    def __str__(self):
        return f""

if __name__ == '__main__':
    api = BicycleController()
    while True:
        menu = Common.menu(["종료","시각화","모델링","머신러닝","배포"])
        if menu == "0": break
        elif menu == "1":
            print(" ### 시각화 ### ")
            model = BicycleModel()
            mtx = model.new_model("Imported table-따릉이_train.csv")
            print(mtx)
            print((f' Train type: {type(mtx)}'))
            print(f' Train columns: {mtx.columns}')
            print(f' Train head: {mtx.head}')
            print(f' Train null의 개수: {mtx.isnull().sum()}')
            # ['date_time', 'wind_direction', 'sky_condition', 'precipitation_form',
            #        'wind_speed', 'humidity', 'low_temp', 'high_temp',
            #        'Precipitation_Probability', 'number_of_rentals']
        elif menu == "2":
            print(" ### 데이터 처리 ### ")
        elif menu == "3":
            print(" ### 머신러닝 ### ")
        elif menu == "4":
            print(" ### 배포 ### ")
        else:
            print(" ### 해당 메뉴 없음 ### ")