import pandas as pd
import acquire
import sklearn.impute
import sklearn.model_selection
import sklearn.preprocessing



def encode_species(train,test):
    encoder = sklearn.preprocessing.OneHotEncoder()
    encoder.fit(train[['species']])

    cols = [ c for c in encoder.categories_[0]]

    m = encoder.transform(train[['species']]).todense()
    train = pd.concat([
        train,
        pd.DataFrame(m, columns=cols, index=train.index)
    ], axis=1).drop(columns='species')

    m = encoder.transform(test[['species']]).todense()
    test = pd.concat([
        test,
        pd.DataFrame(m, columns=cols, index=test.index)
    ], axis=1).drop(columns='species')
    
    return train,test



def prep_iris():
    iris_df = acquire.get_iris_data()
    iris_df = iris_df.drop(columns= ['species_id', 'measurement_id'])
    iris_df = iris_df.rename(columns = {'species_name': 'species'})
    train, test = sklearn.model_selection.train_test_split(iris_df, random_state=123, train_size=.8)
    train, test = encode_species(train,test)
    
    return train, test


def prep_titanic():
    df = acquire.get_titanic_data()
    df = df.drop(columns=['deck', 'class','embark_town'])
    df.embarked = df.embarked.fillna('S')
    df.embarked = df.embarked.astype("|S")
    encoder = sklearn.preprocessing.LabelEncoder()
    df.embarked = encoder.fit_transform(df.embarked)
    df.age = sklearn.preprocessing.MinMaxScaler().fit_transform(df[['age']])
    df.fare = sklearn.preprocessing.MinMaxScaler().fit_transform(df[['fare']])
    avg_age = df.age.mean()
    df.age = df.age.fillna(avg_age)
    
    return df


