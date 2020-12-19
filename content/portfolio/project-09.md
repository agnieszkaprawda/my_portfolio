---
category:
- Machine Learning in Python
date: "2020-12-13T15:44:46+06:00"
image: images/projects/titanic1.jpg
project_images:
title: Predicting Survival of Titanic Passengers in Python
type: portfolio
---

# IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier   # Multilayer Perceptron
from sklearn.neighbors import KNeighborsClassifier # K Nearest Neighbours
from sklearn.svm import SVC                        # Support Vector Machines
from sklearn.gaussian_process import GaussianProcessClassifier # Gaussian Process
from sklearn.naive_bayes import GaussianNB         # Naive Bayes
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis 

from warnings import filterwarnings
filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn import preprocessing


## LOAD AND EXAMINE DATA

df = pd.read_csv('titanic.csv')

# Explore the data

# Check top 10 rows of the dataframe
df.head(10) 
'''
   pclass  survived  ...    cabin embarked
0       1         1  ...       B5        S
1       1         1  ...  C22 C26        S
2       1         0  ...  C22 C26        S
3       1         0  ...  C22 C26        S
4       1         0  ...  C22 C26        S
5       1         1  ...      E12        S
6       1         1  ...       D7        S
7       1         0  ...      A36        S
8       1         1  ...     C101        S
9       1         0  ...      NaN        C

[10 rows x 11 columns]
'''

# Display names of the columns
df.columns
'''
Index(['pclass', 'survived', 'name', 'sex', 'age', 'sibsp', 'parch', 'ticket', 'fare', 'cabin', 'embarked'], 
      dtype='object')

'''

# Display data types of each column
df.dtypes
'''
pclass        int64
survived      int64
name         object
sex          object
age         float64
sibsp         int64
parch         int64
ticket       object
fare        float64
cabin        object
embarked     object
dtype: object
'''

# Summary statistics
# Maximum values for all the columns
maxi = df.max()
print('The maximum values for each column are:\n' + str(maxi))
'''
The maximum values for each column are:
pclass                                3
survived                              1
name        van Melkebeke, Mr. Philemon
sex                                male
age                                  80
sibsp                                 8
parch                                 9
ticket                        WE/P 5735
fare                            512.329
dtype: object
'''

# Minimum values for all the columns
mini = df.min()
print('The minimum values for each column are:\n' + str(mini))
'''
The minimum values for each column are:
pclass                        1
survived                      0
name        Abbing, Mr. Anthony
sex                      female
age                      0.1667
sibsp                         0
parch                         0
ticket                   110152
fare                          0
dtype: object
'''
# Search for NaN values in each column
df.isnull().sum()

'''
pclass         0
survived       0
name           0
sex            0
age          263
sibsp          0
parch          0
ticket         0
fare           1
cabin       1014
embarked       2
dtype: int64

The most missing values are locaed in columns 'age' (263) and 'cabin' (1014).
I will later take care of them using interpolation.
'''


# Drop two columns that will not be useful for the model - sibsp & ticket
df.drop(['ticket', 'embarked'], axis = 1, inplace = True)

# Examine survival based on gender

percGenderSurvived = df.groupby(['sex'])['survived'].sum().transform(lambda x: x/x.sum()).copy()  
print("Percentage of passengers survived based on their gender is as follows:\n" + str(percGenderSurvived))

'''
Percentage of passengers survived based on their gender is as follows:
sex
female    0.678
male      0.322
Name: survived, dtype: float64
'''
# DATA PRE-PROCESSING
# Encode 'sex' column to be female - 0, and male - 1 

# Display top 5 rows of the column
df.sex.head(5)
'''
0    female
1      male
2    female
3      male
4    female
Name: sex, dtype: object
'''
# Transformation to bool type
df.sex = df.sex == "male"

 Display top 5 rows of the column for confirmation
df.sex.head(5)
'''
0    False
1     True
2    False
3     True
4    False
Name: sex, dtype: bool
'''

# Extract only titles from names

# Disply top 5 rows of the column 'name'
df.name.head(5)

# Extract the titles and display unique values
titles = df.name.str.extract(pat = '([A-Za-z]+)\.').copy()
titles = np.unique(titles)
print(titles)
'''
array(['Capt', 'Col', 'Countess', 'Don', 'Dona', 'Dr', 'Jonkheer', 'Lady',
       'Major', 'Master', 'Miss', 'Mlle', 'Mme', 'Mr', 'Mrs', 'Ms', 'Rev',
       'Sir'], dtype=object)
'''
# Check for titles type
type(titles)
'''
numpy.ndarray
'''
# Overwrite 'name' column values
df.name = df.name.str.extract(pat = '([A-Za-z]+)\.')

# Disply 5 rows of the column 'name' for confirmation
df.name.head(5)
'''
0      Miss
1    Master
2      Miss
3        Mr
4       Mrs
Name: name, dtype: object
'''

# Display survival of differently titled individuals
fig = plt.figure(figsize=(12,8))
counter = 1
col = ['blue','orange']
for titles in df['name'].unique():
    fig.add_subplot(3, 6, counter)
    plt.title('Title : {}'.format(titles))
    s = df.survived[df['name'] == titles].value_counts() # series object
    if len(s) > 1: 
        s.sort_index().plot(kind = 'pie', colors = col)
    else: 
        i = s.index[0]
        s.sort_index().plot(kind = 'pie', colors = [col[i]])
    counter += 1
  
# Indepth look at the class survival
survivalTitles = s = df.groupby(['name', 'survived']).agg({'survived': 'count'})
sirvivalinTitle = df.groupby(['name']).agg({'survived': 'count'})
finalTitles = survivalTitles.div(sirvivalinTitle, level='name') * 100
'''

name     survived            
Capt     0         100.000000
Col      0          50.000000
         1          50.000000
Countess 1         100.000000
Don      0         100.000000
Dona     1         100.000000
Dr       0          50.000000
         1          50.000000
Jonkheer 0         100.000000
Lady     1         100.000000
Major    0          50.000000
         1          50.000000
Master   0          49.180328
         1          50.819672
Miss     0          32.307692
         1          67.692308
Mlle     1         100.000000
Mme      1         100.000000
Mr       0          83.751651
         1          16.248349
Mrs      0          21.319797
         1          78.680203
Ms       0          50.000000
         1          50.000000
Rev      0         100.000000
Sir      1         100.000000

'''
# Interpolate missing age entries in the ‘age’ column.
gp = df.groupby('name') #Group the data by title
val = gp.transform('median').age #Find the median value for each title
df['age'].fillna(val, inplace = True) #Fill in missing values



# Check if the age missing values were fixed
df.isnull().sum()

'''
pclass         0
survived       0
name           0
sex            0
age            0
sibsp          0
parch          0
ticket         0
fare           1
cabin       1014
embarked       2
dtype: int64
'''

# Change titles to numbers

# Check the median of age by title
df.groupby('name').age.median()
'''
name
Capt        70.0
Col         54.5
Countess    33.0
Don         40.0
Dona        39.0
Dr          49.0
Jonkheer    38.0
Lady        48.0
Major       48.5
Master       4.0
Miss        22.0
Mlle        24.0
Mme         24.0
Mr          29.0
Mrs         35.5
Ms          28.0
Rev         41.5
Sir         49.0
Name: age, dtype: float64
'''
# Count hoe many passengers hold each title
df.groupby(['name'])['name'].count()
'''
name
Capt          1 -----------> 0 Capitan is much older than all the rest of the crew
Col           4 -----------> 7 Army title
Countess      1 -----------> 3 Royal female
Don           1 -----------> 6 Mature male
Dona          1 -----------> 3 Royal female
Dr            8 -----------> 6 Matue male
Jonkheer      1 -----------> 1 Member of the crew
Lady          1 -----------> 3 Royal female
Major         2 -----------> 7 Army title
Master       61 -----------> 5 Master
Miss        260 -----------> 2 Young female probably unmarried
Mlle          2 -----------> 2 Young female probably unmarried
Mme           1 -----------> 2 Young female probably unmarried
Mr          757 -----------> 4 Mister
Mrs         197 -----------> 8 Married female
Ms            2 -----------> 2 Young female probably unmarried
Rev           8 -----------> 1 Member of the crew
Sir           1 -----------> 6  Mature male
Name: name, dtype: int64

To sum up:
    0 - Capitan
    1 - Other crew members
    2 - Miss + unmarried young females
    3 - Royal female
    4 - Mr
    5 - Master
    6 - 7 out of 8 Dr ( 1 female) + mature males Sir & Don
    7 - Army title
    8 - Mrs + female Dr
'''

# Change titles to numerical values
    
df['name'] = df['name'].replace(['Capt'],0)
df['name'] = df['name'].replace(['Rev','Jonkheer'],1)
df['name'] = df['name'].replace(['Miss','Mlle','Mme','Ms' ],2)
df['name'] = df['name'].replace(['Lady','Dona', 'Countess'],3)
df['name'] = df['name'].replace(['Mr'],4)
df['name'] = df['name'].replace(['Master'],5)
df['name'] = df['name'].replace(['Sir', 'Don', 'Dr'],6)
df['name'] = df['name'].replace(['Major','Col'],7)
df['name'] = df['name'].replace(['Mrs'],8)
# Checking for female Doctor
df.loc[df['name'] == 6, 'sex']
'''
40     True
93     True
100    True
119    True
181    False
206    True
278    True
299    True
508    True
525    True
Name: sex, dtype: bool
'''
df['name'][181] = 8


# Check unique values for titles
df.name.unique()
'''
array([2, 5, 4, 8, 7, 6, 0, 3, 1])
'''
      

## Interpolate missing ticket fare
gp = df.groupby('pclass') #Group the data by class
val = gp.transform('median').fare #Find the median value for each title
df['fare'].fillna(val, inplace = True) #Fill in missing values

# Check if the ticket fare missing values were fixed
df.isnull().sum()
'''
pclass         0
survived       0
name           0
sex            0
age            0
sibsp          0
parch          0
fare           0
cabin       1014
dtype: int64
'''


# Replace 'cabin' identification by only one letter
df.cabin.unique()

cabinClass= df.cabin.str.extract(pat = '([A-Z])').copy()
print(cabinClass)

# Overwrite 'name' column values
df.cabin = df.cabin.str.extract(pat = '([A-Z])')

# Check the unique values for now
df.cabin.unique()
'''
array(['B', 'C', 'E', 'D', 'A', nan, 'T', 'F', 'G'], dtype=object)
'''

# Fill the missing values with 'Z'
df.cabin = df['cabin'].fillna(value = 'Z')


# Check the unique values after filling missing values
df.cabin.unique()
'''
array(['B', 'C', 'E', 'D', 'A', 'Z', 'T', 'F', 'G'], dtype=object)
'''

# Change to numeric values
df['cabin'] = LabelEncoder().fit_transform(df['cabin'].astype(str))

# Check the unique values
df['cabin'].unique()
'''
array([1, 2, 4, 3, 0, 8, 7, 5, 6])
'''
# Numeric values

# Examine the unique values in the 'sibsp' column
df.sibsp.unique()
'''
array([0, 1, 2, 3, 4, 5, 8])

'''
# Only numeric values


# Make sure there are not any more missing values in the dataframe
df.isnull().sum()
'''
pclass      0
survived    0
name        0
sex         0
age         0
sibsp       0
parch       0
fare        0
cabin       0
dtype: int64
'''



# Make sure there are not any more missing values in the dataframe
df.isnull().sum()
'''
pclass      0
survived    0
name        0
sex         0
age         0
sibsp       0
parch       0
fare        0
cabin       0
dtype: int64
'''


# No more missing values in the data set

## machine learning algo training and testing
# seed random number generator for reproducible results
random.seed(1234)

# Split the data into features and label (true outcome, i.e. survived)
label = df['survived'] #initialise feature
feature = df.drop(['survived'], axis=1)  #initalise feature
# Sanity check
label
'''
       1
1       1
2       0
3       0
4       0
       ..
1304    0
1305    0
1306    0
1307    0
1308    0
Name: survived, Length: 1309, dtype: int64
'''
feature.columns
'''
Index(['pclass', 'name', 'sex', 'age', 'sibsp', 'parch', 'fare', 'cabin'], dtype='object')
'''
# Split the data & make sure it is randomised (shuffle=True)
random.seed(1234)
X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size = 0.25,shuffle = True)

# Scale the data
X_train_scaled = preprocessing.scale(X_train, with_mean = True, with_std = True)
scaler = preprocessing.StandardScaler().fit(X_train) #scaler to sclae the test data as well

# Sanity check
X_train_scaled[0]
'''
array([ 0.8273289 , -0.12183624,  0.73980985, -0.02862179, -0.47325356,
       -0.45944255, -0.49725754,  0.50891413])
'''

# Standardise X_test
X_test_scaled = scaler.transform(X_test)

X_test_scaled[0]
'''
array([ 0.8273289 , -1.21836239, -1.35169869, -0.55685232,  0.4247382 ,
       -0.45944255, -0.34078308,  0.50891413])
'''

# specify models as elements of a list
models = [MLPClassifier(), 
          KNeighborsClassifier(n_neighbors = 5), 
          SVC(kernel = 'poly', gamma = 'auto', degree = 5),
          GaussianProcessClassifier(),
          GaussianNB(),
          QuadraticDiscriminantAnalysis()]


# loop over models, train and test
random.seed(1234)
model=[]
for model in models:
    model.fit(X_train_scaled, y_train)
    score = model.score(X_test_scaled, y_test)
    print('Test Set Score:', '%.4f' % score)
'''
MLP:     Test Set Score: 0.8323
KNN:     Test Set Score: 0.7805
SVC:     Test Set Score: 0.7744
GPC:     Test Set Score: 0.8262
GNB:     Test Set Score: 0.7744
QDA:     Test Set Score: 0.7988
'''



