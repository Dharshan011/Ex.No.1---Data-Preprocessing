# Ex.No.1---Data-Preprocessing
## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

##REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
1.Importing the libraries
2.Importing the dataset
3.Taking care of missing data
4.Encoding categorical data
5.Normalizing the data
6.Splitting the data into test and train

## PROGRAM:
```


import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


df=pd.read_csv('Churn_Modelling.csv')
df


df.isnull().sum()


df.duplicated()


df.drop('RowNumber',axis=1,inplace=True)
df.drop('CustomerId',axis=1,inplace=True)
df.drop('Age',axis=1,inplace=True)
df.drop('Gender',axis=1,inplace=True)
df.drop('Surname',axis=1,inplace=True)
df.drop('Geography',axis=1,inplace=True)
df


ms=MinMaxScaler()
df2=pd.DataFrame(ms.fit_transform(df))
df2


X=df2.iloc[:,:-1].values
X


y=df2.iloc[:,-1].values
y


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train)
print("X_train: ",len(X_train))
print(X_test)
print("Size of X_test: ",len(X_test))

```

## OUTPUT:


![Screenshot 2023-09-23 131918](https://github.com/Dharshan011/Ex.No.1---Data-Preprocessing/assets/113497491/3283ed48-add7-4834-97fb-190e34b948fe)


![Screenshot 2023-09-23 131945](https://github.com/Dharshan011/Ex.No.1---Data-Preprocessing/assets/113497491/111d0a59-3273-4851-9ef6-b40b0f66c2e3)


![Screenshot 2023-09-23 131959](https://github.com/Dharshan011/Ex.No.1---Data-Preprocessing/assets/113497491/f9097497-df0b-46ea-815c-8d70d2603b60)


![Screenshot 2023-09-23 132010](https://github.com/Dharshan011/Ex.No.1---Data-Preprocessing/assets/113497491/f5283eef-a0ce-4f27-8996-614d8f2e07fe)


![Screenshot 2023-09-23 132024](https://github.com/Dharshan011/Ex.No.1---Data-Preprocessing/assets/113497491/779dcd93-a783-4a11-ba67-5c080820659d)


![Screenshot 2023-09-23 132045](https://github.com/Dharshan011/Ex.No.1---Data-Preprocessing/assets/113497491/485a705c-5506-4351-928c-e04008b91d98)

![Screenshot 2023-09-23 132039](https://github.com/Dharshan011/Ex.No.1---Data-Preprocessing/assets/113497491/600c4f33-e440-4203-a88b-2bd4fd539e02)

## RESULT


The data set downloaded from kaggle is successfully processed.
