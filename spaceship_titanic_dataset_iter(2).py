#This is iteration 2 of predicting the passengeres teleported to another dimension from spaceship titanic dataset
#Trying to improve the predictions using some different techniques 

import pandas as pd 
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score
from sklearn.metrics import accuracy_score


train_path = "C:/Users/Adwait Tagalpallewar/Downloads/spaceship-titanic/train.csv"
df = pd.read_csv(train_path)


#--------------------
'''Feature Construction'''
#--------------------

#Old features from iter(1):
df[["Deck", "Cabin_num", "Side"]] = df["Cabin"].str.split('/',expand=True) 
df = df.drop('Cabin',axis=1)
df["Total_spend"] = df[["RoomService","ShoppingMall", "FoodCourt", "Spa", "VRDeck"]].sum(axis=1)
df["Has_spent"] = (df["Total_spend"] > 0).astype(int)

#Introducting new features with high signal low gain:
#Luxury ratio = (Spa + VRDeck) / Total_spend
df['Luxury_ratio'] = df[["Spa","VRDeck"]].sum(axis=1) / (df["Total_spend"] +1) # +1 to avoid dividing by 0
#Food ratio = FoodCourt / Total_spend
df["Food_ratio"] = df["FoodCourt"] / (df["Total_spend"] +1)
#Group_size can be gathered from PassengerId since its in form gggg_pp and from Group_size we can see if the passenger is traveling solo
df["Group"] = df["PassengerId"].str.split("_").str[0]
df["Group_size"] = df.groupby("Group")["Group"].transform("count")
df["Is_solo"] = (df["Group_size"] == 1).astype(int)

#dropping useless columns now
useless_cols = ['PassengerId','Name','Group']
df = df.drop(columns = useless_cols)
df.info()

#seperating Label and features
X = df.drop("Transported",axis=1)
y = df["Transported"] 

#splitting into train/test split no validation split since we are using CV
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)


#----------------
'''Data preprocessing'''
#----------------
#seperating columns into 4 groups since every group reqires different preproccessing 
num_cols = ['Age','RoomService','FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Total_spend', 'Luxury_ratio','Food_ratio'] #scale and impute with median 
cat_cols = ['HomePlanet', 'Deck','Cabin_num', 'Side'] #impute with most_frequent and apply onehotencoding
bool_cols = ['CryoSleep','VIP','Has_spent','Is_solo'] #impute with most_frequent 
count_cols = ["Group_size"] #impute with median 

#pipelines
num_pipe = Pipeline([
    ('impute',SimpleImputer(strategy='median')),
    ('scale',StandardScaler()),
])
cat_pipe = Pipeline([
    ('impute',SimpleImputer(strategy='most_frequent')),
    ('encode',OneHotEncoder(handle_unknown='ignore'))
])
bool_pipe = Pipeline([
    ('impute',SimpleImputer(strategy='most_frequent'))
])
count_pipe = Pipeline([
    ('impute',SimpleImputer(strategy='median'))
])

preprocessor = ColumnTransformer(
    transformers = [
        ("num", num_pipe, num_cols),
        ("bool", bool_pipe, bool_cols),
        ("cat", cat_pipe, cat_cols),
        ("count", count_pipe, count_cols)
    ],
    remainder='drop' #we already dropped the useless columns but still it helps, the other value accepted for remainder is 'passthrough' which allows the columns not entering teransformers to pass through as well 
)


#--------------
'''Feautre Selection'''
#--------------
#we will run tests using 2 models to see which performes better logistic regression or randomforest so we will be using same models for feature selection

lr_feature_selector = SelectFromModel(
    LogisticRegression(
        penalty="l1", #l1 regulation take the non important features importance down to 0
        solver="saga",
        C=0.1,
        max_iter=3000,
        random_state=42,
        n_jobs=-1
    ),
    threshold='median' #selects features with importance more than median
)

rf_feature_selector = SelectFromModel(
    RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    ),
    threshold='median'
)


#-------------
'''Final pipelines'''
#-------------
#3 pipeline 1 for rf with rf_feature_selector and another for lr with lr_feature_seelecrtor and last one for rf with lr_feature_selector to see which one performs better 

RF_lr_pipeline = Pipeline([
    ('prep',preprocessor),
    ('feature selection',lr_feature_selector),
    ('model',RandomForestClassifier(
        n_estimators = 350,
        random_state=42,
        n_jobs=-1
    ))
])

RF_rf_pipeline = Pipeline([
    ('prep',preprocessor),
    ('feature selection',rf_feature_selector),
    ('model',RandomForestClassifier(
        n_estimators = 350,
        random_state=42,
        n_jobs=-1
    ))
])

LR_lr_pipeline = Pipeline([
    ('prep',preprocessor),
    ('feature selection',lr_feature_selector),
    ('model',LogisticRegression(
        max_iter=3000,
        random_state=42,
        n_jobs=-1
    ))
])

LR_rf_pipeline = Pipeline([
    ('prep',preprocessor),
    ('feature selection',rf_feature_selector),
    ('model',LogisticRegression(
        max_iter=3000,
        random_state=42,
        n_jobs=-1
    ))
])



#------------
'''Cross-Validation'''
#------------

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

score_RF_lr = cross_val_score(
    RF_lr_pipeline,
    X_train, y_train,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1
)

score_RF_rf = cross_val_score(
    RF_rf_pipeline,
    X_train, y_train,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1
)

score_LR_lr = cross_val_score(
    LR_lr_pipeline,
    X_train, y_train,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1
)

score_LR_rf = cross_val_score(
    LR_rf_pipeline,
    X_train, y_train,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1
)

print('for RF_lr: ', score_RF_lr.mean(), score_RF_lr.std())
print('for RF_rf: ', score_RF_rf.mean(), score_RF_rf.std())
print('for LR_lr: ', score_LR_lr.mean(), score_LR_lr.std())
print('for LR_rf: ', score_LR_rf.mean(), score_LR_rf.std())

'''Results:
for RF_lr:  0.8004033121453951 0.004924980521278633
for RF_rf:  0.7874604988906071 0.004053787393248861
for LR_lr:  0.7874603954507136 0.003929821407228263
for LR_rf:  0.7874594644916705 0.005276639209841899
Clear winner is RF with LR for feature selection, lets see which features hold importance and more info'''

#Lets validate the pipeline on X_test though its already validated using CV
RF_lr_pipeline.fit(X_train,y_train)
y_pred = RF_lr_pipeline.predict(X_test)
print('accuracy score on X_test:',accuracy_score(y_test,y_pred)) #0.796 a little lesser than our CV results but the std is the explaination for this; its normal no drop here 

'''Something to learn from this was, adding new features wont always help the model, also learned a few new techniques like CV and feature_selection '''


#---------------
'''Using test set to make final predictions for iteration(2)'''
#---------------
#importing test dataset
test_path = "C:/Users/Adwait Tagalpallewar/Downloads/spaceship-titanic/test.csv"
test_df = pd.read_csv(test_path)

#creating deatures we created in train df
test_df[["Deck", "Cabin_num", "Side"]] = test_df["Cabin"].str.split('/',expand=True) 
test_df = test_df.drop('Cabin',axis=1)
test_df["Total_spend"] = test_df[["RoomService","ShoppingMall", "FoodCourt", "Spa", "VRDeck"]].sum(axis=1)
test_df["Has_spent"] = (test_df["Total_spend"] > 0).astype(int)
test_df['Luxury_ratio'] = test_df[["Spa","VRDeck"]].sum(axis=1) / (test_df["Total_spend"] +1) 
test_df["Food_ratio"] = test_df["FoodCourt"] / (test_df["Total_spend"] +1)
test_df["Group"] = test_df["PassengerId"].str.split("_").str[0]
test_df["Group_size"] = test_df.groupby("Group")["Group"].transform("count")
test_df["Is_solo"] = (test_df["Group_size"] == 1).astype(int)
useless_cols = ['PassengerId','Name','Group']
X_testdataset = test_df.drop(columns = useless_cols)

#predicting on RF_lr_pipeline
RF_lr_pipeline.fit(X,y)
y_pred = RF_lr_pipeline.predict(X_testdataset)

submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    'Transported': y_pred
})

submission.to_csv('submission_iter(2).csv',index=False) ##0.7996 accuracy score on kaggle; LB rank = 1453/2692, improvement from last time not bad ig it wasnt all worth nothing afterall

#Aint gonna lie this 0.001 gained honestly is worth more than 0.01 gained by luck to me rn, im super pumped up!!!!
