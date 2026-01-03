#goal: Predict which passengers are transported to an alternate dimension 
#the dataset used belongs to kaggle competition 'spaceship titanic'
#will try to compare multiple classification models to achive maximum accuracy scores(the competition is judging on the basis of accuracy scores)
#while mintaining a balanced treadoff between precissiona and recall 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from collections import Counter
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder,FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import precision_score,accuracy_score,recall_score,f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


train_path = "C:/Users/Adwait Tagalpallewar/Downloads/spaceship-titanic/train.csv"
df = pd.read_csv(train_path)
print(df.head())
print(df.columns) #there seems to be many null values cant remove them all so we'll try to fill the values with most freq for object types and median for continious valued columns 
print(df.isna().sum())
df.info()
print(df.describe())

print(df.isnull().mean().sort_values(ascending=False)) #missingness seems to be low(2-2.5%) can consider dropping but lets impute these values in case we get some value from them 

#There are some features that identify each row uniquely so will have to remove them 
useless_col = ['PassengerId','Name']
df = df.drop(useless_col,axis=1)


#-----------------
'''Feature Construction'''
#-----------------

#The column 'Cabin' consistes of values a/b/c: deck/num/side these featuers can be formed from te feature Cabin
df[["Deck", "Cabin_num", "Side"]] = df["Cabin"].str.split('/',expand=True) 
df = df.drop('Cabin',axis=1)
#Total_spend can be formed using all the columns showcasing spend on the boat
df["Total_spend"] = df[["RoomService","ShoppingMall", "FoodCourt", "Spa", "VRDeck"]].sum(axis=1)
#Has_spent can be derived from Total_spend; if Total_spend > 0, Has_spent = 1 another categorical feature 
df["Has_spent"] = (df["Total_spend"] > 0).astype(int)


#feature distribution
df.hist(figsize=(12,4))
plt.tight_layout()
plt.show() #shows that scaling is required also the spending features are right skewed which might cause negative impact on model so gotta handle that too 

#class distribution in the target throughout the dataset
counts = Counter(df['Transported'])
plt.bar(counts.keys(),counts.values(),width=0.3)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class distribution in target')
plt.show() #balanced class distribution  

#seperating target and features
X = df.drop('Transported',axis=1)
y = df['Transported']

#preprocessing and pipelining data 
num_cols = ["Age", "RoomService", "FoodCourt", "Spa","ShoppingMall", "VRDeck","Total_spend"]
bool_cols = ["CryoSleep", "VIP","Has_spent"]
obj_cols = ["HomePlanet", "Destination","Deck", "Cabin_num", "Side"]


num_pipeline = Pipeline([
    ('impute',SimpleImputer(strategy='constant',fill_value=0)),
    ("log", FunctionTransformer(np.log1p)), #helps in reducing the skewness of data as we noticed in histograms for spending features 
    ('scaling',StandardScaler())
])

bool_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
])

obj_pipeline = Pipeline([
    ('impute',SimpleImputer(strategy='constant',fill_value='Missing')),
    ('encoding',OneHotEncoder(handle_unknown="ignore"))
])

preprocessing = ColumnTransformer([
    ('num',num_pipeline,num_cols),
    ('bool',bool_pipeline,bool_cols),
    ('obj',obj_pipeline,obj_cols)
])


#lets define a function that will automate the preprocessing-->trainin-->testing-->evaluation for provided model and data 
def train_and_evaluate(model,train_features,train_targets,test_features,test_targets=None):
    
    pipeline = Pipeline([
        ('prep',preprocessing),
        ('model',model),
    ])

    pipeline.fit(train_features,train_targets)
    y_pred = pipeline.predict(test_features)
    
    metrics = None
    if test_targets is not None:
        metrics = {
            "accuracy": accuracy_score(test_targets, y_pred),
            "precision": precision_score(test_targets, y_pred),
            "recall": recall_score(test_targets, y_pred),
            "f1": f1_score(test_targets, y_pred)
        }

    return pipeline,y_pred,metrics


'''I automated the process of training and evaluation because I faced an issue of writting the pipeline again and again everytime for every model in my last project,
so that was something I wanted to make sure doesnt happen again(learnt smth from my past experience xD)'''


#----------------
'''Model Selection'''
#----------------
#lets train two baseline models and compare the results on validation set and see which model to go ahead with 

#splitting data into train/validation
X_train,X_validate,y_train,y_validate = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

#Baseline_model(1) Logistic Regression
model1 = LogisticRegression(n_jobs=-1,max_iter=500,random_state=42)

#Baseline_model(2) RandomForestClassifier
model2 = RandomForestClassifier(n_estimators=350,n_jobs=-1)


#testing on both models
pipeline1,y_pred1,metrics1 = train_and_evaluate(model1,X_train,y_train,X_validate,y_validate) #logistic regression baseline

pipeline2,y_pred2,metrics2 = train_and_evaluate(model2,X_train,y_train,X_validate,y_validate) #randomforestclassifier baseline

print(f'evaluation on validation data for model1: {metrics1} \n evaluation on validation data for model2: {metrics2}')

'''Results:
evaluation on validation data for model1: {'accuracy': 0.7832087406555491, 'precision': 0.778149386845039, 'recall': 0.7968036529680366, 'f1': 0.787366046249295} 
 evaluation on validation data for model2: {'accuracy': 0.8010350776308223, 'precision': 0.8223844282238443, 'recall': 0.771689497716895, 'f1': 0.7962308598351001}'''

'''On watching closely accuracy and precision are better in RF while recall is greater in LR thats the tread-off so we'd decide from f1 score which model to choose 
RF has slight edge in f1 score over LR so choosing RandomForest for hypertuning also if u noticed the features constructed from our data were informative'''


#----------------
'''Hypertuning our baseline RF model'''
#----------------
#Hypertuning is important to see how much better performance can we get out of our model by tuning some parameters 

#Tuning n_estimators (greater n less overfitting)
values_n = [250,300,350,400,450,500,550,600,650,700]
for n in values_n:
    rf = RandomForestClassifier(n_estimators=n, n_jobs=-1)
    pipeline,y_pred,metrics= train_and_evaluate(rf,X_train,y_train,X_validate,y_validate)
    plt.scatter(n,metrics['accuracy'],color='blue')

plt.xlabel('n')
plt.ylabel('accuracy')
plt.show() #locking n=300 cuz thats the only value which isnt fluctuiting much for changing random states so i consider it a stable value

#Letting max_depth be at None tuning it isnt helping; ig we'll have to let the tree grow xD

#Tuning min_samples_leaf
values_minsamplesleaf = range(1,50)
for n in values_minsamplesleaf:
    rf = RandomForestClassifier(n_estimators=300,min_samples_leaf=n, n_jobs=-1)
    pipeline,y_pred,metrics= train_and_evaluate(rf,X_train,y_train,X_validate,y_validate)
    plt.scatter(n,metrics['accuracy'],color='blue')

plt.xlabel('n')
plt.ylabel('accuracy')
plt.show() #locking min_samples_leaf=None, tuning is not helping  


#----------------
'''First finalized model'''
#----------------
model = RandomForestClassifier(n_estimators=300, n_jobs=-1)

#importing test data and preparing it to predict values for first submission 
test_path = "C:/Users/Adwait Tagalpallewar/Downloads/spaceship-titanic/test.csv"
test_df = pd.read_csv(test_path)

X_test = test_df.drop(useless_col,axis=1)

X_test[["Deck", "Cabin_num", "Side"]] = X_test["Cabin"].str.split('/',expand=True) 
X_test = X_test.drop('Cabin',axis=1)
X_test["Total_spend"] = X_test[["RoomService","ShoppingMall", "FoodCourt", "Spa", "VRDeck"]].sum(axis=1)
X_test["Has_spent"] = (X_test["Total_spend"] > 0).astype(int)


pipeline,y_pred,metrics = train_and_evaluate(model,X,y,X_test)

submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Transported': y_pred
})

submission.to_csv('submissionRF.csv',index=False) #0.798 accuracy score on kaggle; LB rank = 1522/2692

'''Learned a lot from this 1st iteration, stuff like feature creation and how to create informative features which help model predict better 
Improvements for next iteration : 1)Try to explore data for more informative features that might help model,
                                  2)Proper Feature selection using feature selection tool,
                                  3)Try XGBoosting,Cross-Validation,
                                  4)reach atleast 80-81% accuracy on test dataset '''