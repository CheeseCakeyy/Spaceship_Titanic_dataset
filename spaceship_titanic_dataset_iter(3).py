#This is iteration 3 of trying to improve results on testset of spaceship titanic comp dataset 
#already tried RF,feature selection and Cross validation now its turn to use XGBoost with Cross validation
#XGBoost doesnt require scaling of data so no scaling of any features this time it will only be imputing this time 

import pandas as pd 
from sklearn.model_selection import cross_val_score,StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


train_path = "data/train.csv"
df = pd.read_csv(train_path)


#--------------------
'''Feature Construction'''
#--------------------
#Old features from iter(1,2):
df[["Deck", "Cabin_num", "Side"]] = df["Cabin"].str.split('/',expand=True) 
df = df.drop('Cabin',axis=1)
df["Total_spend"] = df[["RoomService","ShoppingMall", "FoodCourt", "Spa", "VRDeck"]].sum(axis=1)
df["Has_spent"] = (df["Total_spend"] > 0).astype(int)
df['Luxury_ratio'] = df[["Spa","VRDeck"]].sum(axis=1) / (df["Total_spend"] +1) 
df["Food_ratio"] = df["FoodCourt"] / (df["Total_spend"] +1)
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


#--------------
'''Preprocessing Pipelines'''
#--------------
num_cols = ['Age','RoomService','FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Total_spend', 'Luxury_ratio','Food_ratio',"Group_size"] #impute with median 
cat_cols = ['HomePlanet', 'Deck','Cabin_num', 'Side'] #impute with most_frequent and apply onehotencoding
bool_cols = ['CryoSleep','VIP','Has_spent','Is_solo'] #impute with most_frequent 


num_pipe = Pipeline([
    ('impute',SimpleImputer(strategy='median'))
])
bool_pipe = Pipeline([
    ('impute',SimpleImputer(strategy='most_frequent'))
])
cat_pipe = Pipeline([
    ('impute',SimpleImputer(strategy='most_frequent')),
    ('encode',OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num',num_pipe,num_cols),
        ('cat',cat_pipe,cat_cols),
        ('bool',bool_pipe,bool_cols)
    ],
    remainder='drop'
)


#-------------
'''XGBoost'''
#-------------
#model
xgb_model = XGBClassifier(
    n_estimators = 350,
    learning_rate = 0.03, #as the name of parameter suggests it determines how much to learn from every iteration/tree 
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.9,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

#pipeline
xgb_pipeline = Pipeline([
    ('prep',preprocessor),
    ('model',xgb_model)
])

#cross validation
cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
scores = cross_val_score(
    xgb_pipeline,
    X,
    y,
    cv=cv,
    scoring = 'accuracy',
    n_jobs=-1
)

print("XGBoost CV scores: ", scores)
print("XGBoost CV mean:", scores.mean())
print("XGBoost CV std :", scores.std())

'''After a little hypertuning we got the results as following:
XGBoost CV scores:  [0.81138585 0.79815986 0.8119609  0.81357883 0.78883774]
XGBoost CV mean: 0.8047846367533952
XGBoost CV std : 0.009701582976280928
Honestly which are great so its ready in my opinion to give a shot at testing dataset prediction'''


#------------
'''Iteration 3 submission '''
#------------
test_path = "data/test.csv"
test_df = pd.read_csv(test_path)

#creating features, created in train df
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

#predicting on xgb_pipeline
xgb_pipeline.fit(X,y)
y_pred = xgb_pipeline.predict(X_testdataset)

submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    'Transported': y_pred
})

submission.to_csv('submission_iter(3).csv',index=False) #0.805 accuracy score on kaggle; LB rank = 740/2692


