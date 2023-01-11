from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import sys
from sklearn.datasets import make_blobs
def task(input_file):
    df=pd.read_csv(input_file, index_col=False)
    #df.drop(['drop'],axis=1)
    #DATA CLEANING
    severity_columns = df.filter(like='Severity_').columns
    gender_columns = df.filter(like='Gender_').columns
    contact_columns = df.filter(like='Contact_').columns
    df['Severity_None'].replace({1:'None',0:'No'},inplace =True)
    df['Severity_Mild'].replace({1:'Mild',0:'No'},inplace =True)
    df['Severity_Moderate'].replace({1:'Moderate',0:'No'},inplace =True)
    df['Severity_Severe'].replace({1:'Severe',0:'No'},inplace =True)
    df['Gender_Female'].replace({1:'Female',0:'No'},inplace =True)
    df['Gender_Male'].replace({1:'Male',0:'No'},inplace =True)
    df['Gender_Transgender'].replace({1:'Transgender',0:'No'},inplace =True)
    df['Contact_Dont-Know'].replace({1:'DontKnow',0:'No'},inplace =True)
    df['Contact_No'].replace({1:'NoContact',0:'No'},inplace =True)
    df['Contact_Yes'].replace({1:'Yes',0:'No'},inplace =True)
    df['Condition']=df[severity_columns].values.tolist()
    df['Gender']=df[gender_columns].values.tolist()
    df['Contact']=df[contact_columns].values.tolist()
    def remove(a):
        a = set(a) 
        a.discard("No")
        final = ''.join(a)
        return final
    df['Condition'] = df['Condition'].apply(remove)
    df['Gender'] = df['Gender'].apply(remove)
    df['Contact'] = df['Contact'].apply(remove)
    df=df.drop(['Severity_Severe','Severity_Mild','Severity_Moderate','Severity_None','Gender_Female','Gender_Male','Gender_Transgender','Contact_Dont-Know','Contact_No','Contact_Yes'],axis=1)
    df_task1=df.drop(['Country'],axis=1)
    df_task1['Condition'].replace({'None':0,'Mild':1,'Moderate':2,'Severe':3},inplace =True)
    #TASK-1 
    #DATA CLEANING: Converting 4-class to 2-class problem
    df_task1['Condition'].replace({0:0,1:1,2:1,3:1}, inplace=True)
    df_task1['Gender'].replace({'Male':0,'Female':1,'Transgender':2},inplace =True)
    df_task1['Contact'].replace({'NoContact':0,'DontKnow':1,'Yes':2},inplace =True)
    #MODEL:
    y=df_task1["Condition"]
    X=df_task1.drop(['Condition'],axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    DecisionTree_task1 = DecisionTreeClassifier(max_depth=2,
                                        min_samples_split=10,
                                        random_state=0)
    #MODEL FITTING AND PREDICTING
    DecisionTree_task1.fit(X_train,y_train)

    y_pred=DecisionTree_task1.predict(X_test)
    y_pred
    correct_predictions_1 = 0
    for true, predicted in zip(y_test, y_pred):
        if true == predicted:
            correct_predictions_1 += 1
    accuracy_1 = correct_predictions_1/len(y_test)
    scores_d = cross_val_score(DecisionTree_task1, X, y, cv=5)
    print("TASK-1")
    print('DecisionTreeClassifier Training scores: ', scores_d.mean())
    print("DecisionTreeClassifier prediction accuracy: ",accuracy_1)
    #TASK-2
    #DATA CLEANING:
    df_task2=df.drop(['Country'],axis=1)
    df_task2['Condition'].replace({'None':0,'Mild':1,'Moderate':2,'Severe':3},inplace =True)
    df_task2['Gender'].replace({'Male':0,'Female':1,'Transgender':2},inplace =True)
    df_task2['Contact'].replace({'NoContact':0,'DontKnow':1,'Yes':2},inplace =True)
    #MODEL:
    y2 = df_task2['Condition']
    X2 = df_task2.drop(['Condition'], axis=1)
    X2, y2 = make_blobs(n_samples=500, n_features=19, centers=20,random_state=0)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2,test_size=0.2, random_state=0,stratify=y2)

    DecisionTree_task2 = DecisionTreeClassifier(max_depth=10, 
                                        min_samples_split=2,
                                        random_state=0)

    scores = cross_val_score(DecisionTree_task2, X2, y2, cv=5)
    print("TASK-2")
    print('DecisionTreeClassifier Training score: ', scores.mean())
    #MODEL FITTING and PREDICTING:
    DecisionTree_task2.fit(X_train2,y_train2)
    y_pred_task2=DecisionTree_task2.predict(X_test2)

    correct_predictions_2 = 0
    for true, predicted in zip(y_test2, y_pred_task2):
        if true == predicted:
            correct_predictions_2 += 1
    accuracy_2 = correct_predictions_2/len(y_test2)
    print('DecisionTreeClassifier prediction accuracy: ',accuracy_2)
if __name__ == '__main__':
    input_file= sys.argv[1]
    task(input_file)