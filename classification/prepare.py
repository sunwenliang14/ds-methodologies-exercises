import pandas as pd
import acquire
import sklearn.impute
import sklearn.model_selection
import sklearn.preprocessing
import warnings
warnings.filterwarnings("ignore")



def encode_species(train,test):
    encoder = sklearn.preprocessing.LabelEncoder()
    train['species'] = encoder.fit_transform(train[['species']])
    test['species'] = encoder.fit_transform(test[['species']])
    return train, test

    
def prep_iris():
    iris_df = acquire.get_iris_data()
    iris_df = iris_df.drop(columns= ['species_id', 'measurement_id'])
    iris_df = iris_df.rename(columns = {'species_name': 'species'})
    train, test = sklearn.model_selection.train_test_split(iris_df, random_state=123, train_size=.7)
    train, test = encode_species(train,test)
    
    return train, test





def encode_embarked(train,test):
    encoder = sklearn.preprocessing.LabelEncoder()
    train['embarked'] = encoder.fit_transform(train[['embarked']])
    test['embarked'] = encoder.fit_transform(test[['embarked']])
    return train, test

def scale_age_and_fare(train,test):
    train.age = sklearn.preprocessing.MinMaxScaler().fit_transform(train[['age']])
    test.age = sklearn.preprocessing.MinMaxScaler().fit_transform(test[['age']])
    train.fare = sklearn.preprocessing.MinMaxScaler().fit_transform(train[['fare']])
    test.fare = sklearn.preprocessing.MinMaxScaler().fit_transform(test[['fare']])
    return train, test

def fillna_age(train,test):
    avg_age = train.age.mean()
    train.age = train.age.fillna(avg_age)
    test.age = test.age.fillna(avg_age)
    return train, test


def prep_titanic():
    df = acquire.get_titanic_data()
    df = df.drop(columns=['deck', 'class','embark_town'])
    df.embarked = df.embarked.fillna('S')
    df.embarked = df.embarked.astype("|S")
    train, test = sklearn.model_selection.train_test_split(df, random_state=123, train_size=.8)
    train, test = encode_embarked(train,test)
    train, test = scale_age_and_fare(train,test)
    train, test = fillna_age(train,test)
    
    return train, test 


