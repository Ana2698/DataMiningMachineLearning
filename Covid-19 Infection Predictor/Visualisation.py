#Import Statements
import pandas as pd
import numpy as np
import sys
import seaborn as sns
from IPython.display import display
#!pip install dataframe_image
import dataframe_image as dfi
#!pip install wordcloud
from wordcloud import WordCloud
from wordcloud import ImageColorGenerator
from wordcloud import STOPWORDS
import matplotlib.pyplot as plt
from pylab import rcParams

def analyze_visualisation(file):
    #Reading the dataset
    df=pd.read_csv(file, index_col=False)
    #df=df.drop(['Unnamed: 0'], axis=1)
    #DATA CLEANING
    #Merging related features into one feature
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
    #Dropping the columns
    df=df.drop(['Severity_Severe','Severity_Mild','Severity_Moderate','Severity_None','Gender_Female','Gender_Male','Gender_Transgender','Contact_Dont-Know','Contact_No','Contact_Yes'],axis=1)
    df1=df.drop(['Country'],axis=1)
    df1['Condition'].replace({'None':0,'Mild':1,'Moderate':2,'Severe':3},inplace =True)
    df1['Gender'].replace({'Male':0,'Female':1,'Transgender':2},inplace =True)
    df1['Contact'].replace({'NoContact':0,'DontKnow':1,'Yes':2},inplace =True)
    #ANALYSIS
    df_mode=df1.mode(axis=0) #mode calculation 
    df_sum=df1.sum()#sum calculation 
    df_var=df1.var()#variance calculation
    df_std=df1.std()
    mode_array=[]
    range_array=[]
    mean_array=[]
    sum_array=[]
    var_array=[]
    std_array=[]
    dict_mode=df_mode.to_dict(orient='list')
    dict_sum=df_sum.to_dict()
    dict_var=df_var.to_dict()
    dict_std=df_std.to_dict()
    dict_mode
    for key, value in dict_mode.items():
        mode_array.append(value)
    mode_array 
    for key1, value1 in dict_sum.items():
        sum_array.append(value1)
    sum_array
    for key2, value2 in dict_var.items():
        var_array.append(value2)
    for key3, value3 in dict_std.items():
        std_array.append(value3)
    for (columnName, columnData) in df1.iteritems():
        range_attr ="[{},{}]" .format(df1[columnName].min(),df1[columnName].max()) # range calculation
        mean_attr=df1[columnName].mean() # mean calculation 
        range_array.append(range_attr)
        mean_array.append(mean_attr)
    data={"Features":["Fever","Tiredness","Dry-Cough","Difficulty-in-Breathing","Sore-Throat","None_Sympton","Pains","Nasal-Congestion"
                    ,"Runny-Nose","Diarrhea","None_Experiencing","Age_0-9","Age_10-19","Age_20-24","Age_25-59","Age_60+"
                    ,"Condition","Gender","Contact"],"range":range_array,"mean":mean_array,"mode":mode_array,"sum":sum_array,
                    "variance": var_array,"standard_dev": std_array
         }
        
    stat_df = pd.DataFrame(data)
    dfi.export(stat_df, 'StatisticalCalculations.png')
    display(stat_df)
    #Correlations
    corr=pd.DataFrame(df1.corr())
    #corr.to_png()
    dfi.export(corr,'Correlations.png')
    #VISUALISATION  
    #WORDCLOUD
    text = " ".join(i for i in df.Country)
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
    plt.figure( figsize=(12,8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Country Wise Frequency",fontsize=26)
    plt.show()
    plt.savefig('WordCloud.png')
    #CORRELATION HEATMAP
    corr=pd.DataFrame(df1.corr())
    rcParams['figure.figsize'] = 18, 18
    k = 22
    cols = corr.nlargest(k, 'Condition')['Condition'].index
    cm = np.corrcoef(df1[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
    plt.show()
    #PIE CHART
    Names=df["Condition"].value_counts().index.tolist()
    Sizes=df["Condition"].value_counts().values.tolist()
    color= ['lightcyan','powderblue','steelblue','deepskyblue']
    explode = [0.05 for i in Names]
    plt.figure(figsize= (8,8))
    plt.pie(Sizes, labels = Names, startangle=0, explode =explode,colors = color, shadow = True)
    centre_circle = plt.Circle((0,0),0.70,fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.title('Infection Level Distribution',fontsize = 30)
    plt.axis('equal')  
    plt.tight_layout()
    #HORIZONTAL BAR GRAPH
    sort_by_value = dict(sorted(dict_sum.items(), key=lambda item: item[1]))
    symptoms= list(sort_by_value.keys())
    symptom_counts=list(sort_by_value.values())
    del symptoms[1:7]
    del symptom_counts[1:7]
    del symptoms[10:13]
    del symptom_counts[10:13]
    fig = plt.figure(figsize=(10, 5))
    plt.barh(symptoms, symptom_counts, color='wheat')
    plt.xlabel("count_of_symptoms")
    plt.ylabel("symptoms")
    plt.title("prevelent symptoms")
    plt.show()
    #LINE GRAPH
    age_groups= list(dict_sum.keys())
    age_counts=list(dict_sum.values())
    del age_groups[0:11]
    del age_groups[5:8]
    del age_counts[0:11]
    del age_counts[5:8]
    plt.figure(figsize=(10,10))
    plt.plot(age_groups, age_counts, color='red')
    plt.title('Age Groups Vs Number of people', fontsize=14)
    plt.xlabel('Age_groups', fontsize=14)
    plt.ylabel('Number of people', fontsize=14)
    plt.grid(True)
    plt.show()
    #Distribution
    plt.figure(figsize=(10,10))
    sns.kdeplot(data = stat_df['mean'])
    plt.xlim([-1,2]) 
    plt.title('Distribution of mean metric')
    plt.xlabel("mean of attributes", size=12) 
    plt.ylabel("Frequency", size=12)  
    plt.grid(True, alpha=0.3, linestyle="--")     
    plt.show()
if __name__ == '__main__': # THIS FUNCTION ONLY WORKS FOR TERMINAL/POWERSHELL.
    input_file= sys.argv[1]
    analyze_visualisation(input_file)
    