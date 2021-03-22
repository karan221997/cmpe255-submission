import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

        
class DiabetesClassifier:
    def __init__(self) -> None:
        col_names = ['pregnant', 'glucose', 'bp', 'skin',
                     'insulin', 'bmi', 'pedigree', 'age', 'label']
        self.pima = pd.read_csv('diabetes.csv', header=0,
                                names=col_names, usecols=col_names)
        print(self.pima.head())
        self.X_test = None
        self.y_test = None

    def define_feature(self, list_f):
        # feature_cols = ['pregnant', 'insulin', 'bmi', 'age']
        # X = self.pima[feature_cols]
        X = list_f
        y = self.pima.label
        return X, y

    def train(self, list_f, split):
        # split X and y into training and testing sets
        print(split)
        X, y = self.define_feature(list_f)
        X_train, self.X_test, y_train, self.y_test = train_test_split(
            X, y, test_size=split, random_state=1)
        # train a logistic regression model on the training set
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        return logreg

    def predict(self, list_f, split=.3):

        model = self.train(list_f, split)
        y_pred_class = model.predict(self.X_test)
        return y_pred_class

    def calculate_accuracy(self, result):
        return metrics.accuracy_score(self.y_test, result)

    def examine(self):
        dist = self.y_test.value_counts()
        print(dist)
        percent_of_ones = self.y_test.mean()
        percent_of_zeros = 1 - self.y_test.mean()
        return self.y_test.mean()

    def confusion_matrix(self, result):
        return metrics.confusion_matrix(self.y_test, result)
    
if __name__ == "__main__":
    # In this first iteration i have taken glucose to predict and test size as 0.35
    classifer = DiabetesClassifier()
    df = classifer.pima

    # corr = df.corr()
    # print(corr)

    fetaure_list = df[["glucose"]]
    result = classifer.predict(fetaure_list, 0.35)
    print("----------------------ITERATION 1---------------")
    print(f"Predicition={result}")
    score = classifer.calculate_accuracy(result)
    print(f"score={score}")
    con_matrix = classifer.confusion_matrix(result)
    print(f"confusion_matrix={con_matrix}")
    # score=0.7657992565055762
    # confusion_matrix=[[159  15]
    # [ 48  47]]

    # In this second iteration i have taken glucose,bmi and age as my 3 parameters
    # and taken test size as 0.3
    classifer = DiabetesClassifier()
    df = classifer.pima

    fetaure_list = df[["glucose", "bmi", "age"]]
    result = classifer.predict(fetaure_list, 0.3)
    print("----------------------ITERATION 2---------------")
    print(f"Predicition={result}")
    score = classifer.calculate_accuracy(result)
    print(f"score={score}")
    con_matrix = classifer.confusion_matrix(result)
    print(f"confusion_matrix={con_matrix}")
    # score=0.7835497835497836
    # confusion_matrix=[[132  14]
    #  [ 36  49]]
    

    # In the third iteration i have taken bp,glucose,bmi and replaced the 0 value 
    # with mean of the values and took test size as 0.3
    classifer = DiabetesClassifier()
    df = classifer.pima

    print("----------------------ITERATION 3---------------")
    print("Total bp with 0 value: ", df[df.bp == 0].shape[0]) # 35
    print("Total glucose with 0 value: ", df[df.glucose == 0].shape[0]) #5
    print("Total bmi with 0 value: ", df[df.bmi == 0].shape[0]) #11 values
    mean_glucose = df['glucose'].mean()  # replacing with mean
    df['glucose'] = df['glucose'].replace(
        to_replace=0, value=mean_glucose)
    # 
    # print("Total : ", df[df.glucose == 0].shape[0])

    mean_bp = df['bp'].mean()  # replacing with mean
    df['bp'] = df['bp'].replace(
        to_replace=0, value=mean_bp)
    # print("bp")
    # print("Total : ", df[df.bp == 0].shape[0])

    mean_bmi = df['bmi'].mean()  # replacing with mean
    df['bmi'] = df['bmi'].replace(
        to_replace=0, value=mean_bmi)
    # print("bmi")
    # print("Total : ", df[df.bmi == 0].shape[0])

    fetaure_list = df[['glucose', 'bmi' , 'pregnant']]
    result = classifer.predict(fetaure_list, 0.3)
    print(f"Predicition = {result}")
    score = classifer.calculate_accuracy(result)
    print(f"score = {score}")
    con_matrix = classifer.confusion_matrix(result)
    print(f"confusion_matrix ={con_matrix}")

    # score = 0.8008658008658008
    # confusion_matrix =[[133  13]
    # [ 33  52]]