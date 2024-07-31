# Deeplearning_Use-Case
#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as mpl
from sklearn import preprocessing


# In[5]:


df.shape


# In[6]:


df.isnull()


# In[7]:


df.isnull().sum()


# In[12]:


df.head()


# In[13]:


df.tail(6)


# In[20]:


import seaborn as sns
sns.set(color_codes=True)


# In[15]:


df.info()


# In[23]:


import pandas as pd


# In[1]:


from matplotlib import pyplot as plt


# In[6]:


import pandas as pd
df=pd.read_csv(r"C:\Users\Yuktha\Desktop\shopping_trends_updated.csv")
df


# In[7]:


import matplotlib.pyplot as plt

# Assuming 'column_x' and 'column_y' are the columns you want to plot
plt.plot(df['Gender'], df['Age'])
plt.xlabel('Gender')
plt.ylabel('Age')
plt.title('Gender over Age')
plt.show()


# In[12]:


import matplotlib.pyplot as plt

plt.bar(df['Customer ID'], df['Size'])
plt.xlabel('Customer ID')
plt.ylabel('Size')
plt.title('Customer ID over Size')
plt.show()


# In[2]:


# Python code to illustrate  
# regression using data set 
import matplotlib 
    
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn import datasets, linear_model 
import pandas as pd 
   
# Load CSV and columns 
df = pd.read_csv(r"C:\Users\Yuktha\Desktop\shopping_trends_updated.csv") 
   
Y = df['size'] 
X = df['color'] 
   
X=X.values.reshape(len(X),1) 
Y=Y.values.reshape(len(Y),1) 
   
# Split the data into training/testing sets 
X_train = X[:-250] 
X_test = X[-250:] 
   
# Split the targets into training/testing sets 
Y_train = Y[:-250] 
Y_test = Y[-250:] 
   
# Plot outputs 
plt.scatter(X_test, Y_test,  color='black') 
plt.title('Test Data') 
plt.xlabel('Size') 
plt.ylabel('color') 
plt.xticks(()) 
plt.yticks(()) 
   
  
# Create linear regression object 
regr = linear_model.LinearRegression() 
   
# Train the model using the training sets 
regr.fit(X_train, Y_train) 
   
# Plot outputs 
plt.plot(X_test, regr.predict(X_test), color='red',linewidth=3) 
plt.show() 


# In[3]:


import matplotlib.pyplot as plt

# Assuming 'column_x' and 'column_y' are the columns you want to plot
plt.plot(df['Category'], df['Season'])
plt.xlabel('Category')
plt.ylabel('Season')
plt.title('Category over Season')
plt.show()


# In[2]:


import matplotlib 
    
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn import datasets, linear_model 
import pandas as pd 


# In[3]:


df = pd.read_csv(r"C:\Users\Yuktha\Desktop\shopping_trends_updated.csv") 


# In[4]:


df.head()


# In[15]:


X = df['Purchase Amount (USD)'] 
Y = df['Previous Purchases']  


# In[17]:


Y


# In[ ]:


X=X.values.reshape(len(X),1) 
Y=Y.values.reshape(len(Y),1) 


# In[21]:


X_train = X[:-250] 
X_test = X[-250:] 


# In[22]:


Y_train = Y[:-250] 
Y_test = Y[-250:] 


# In[24]:


plt.scatter(X, Y,  color='black') 
plt.title('Test Data') 
plt.xlabel('Purchase Amount (USD)') 
plt.ylabel('Previous Purchases') 
plt.xticks(()) 
plt.yticks(()) 
   
 
# Create linear regression object 
#regr = linear_model.LinearRegression() 
   
# Train the model using the training sets 
#regr.fit(X_train, Y_train) 
   
# Plot outputs 
#plt.plot(X_test, regr.predict(X_test), color='red',linewidth=3) 
plt.show() 


# In[11]:


X_test


# In[14]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


df=pd.read_csv(r"C:\Users\Yuktha\Desktop\height-weight.csv")


# In[16]:


df.head()


# In[18]:


plt.scatter(df['Weight(Pounds)'],df['Height(Inches)'])
plt.xlabel("Weight")
plt.ylabel("Height")


# In[19]:


df.corr()


# In[20]:


import seaborn as sns
sns.pairplot(df)


# In[23]:


X=df['Weight(Pounds)']
type(df)


# In[24]:


Y=df['Height(Inches)']
type(df)


# In[27]:


X=df[['Weight(Pounds)']]
Y=df['Height(Inches)']


# In[29]:


X_series=df[['Weight(Pounds)']]
np.array(X_series).shape


# In[30]:


Y_series=df['Height(Inches)']
np.array(Y_series).shape


# In[31]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.35,random_state=46)


# In[32]:


X_train.shape


# In[33]:


X_train


# In[34]:


from sklearn.preprocessing import StandardScaler


# In[35]:


scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)


# In[36]:


X_test=scaler.transform(X_test)


# In[37]:


X_test


# In[38]:


from sklearn.linear_model import LinearRegression


# In[39]:


regression=LinearRegression(n_jobs=-1)


# In[40]:


regression.fit(X_train,Y_train)


# In[41]:


print("Coefficient or slope:",regression.coef_)
print("Intercept:",regression.intercept_)


# In[42]:


plt.scatter(X_train,Y_train)


# In[43]:


plt.scatter(X_train,Y_train)
plt.plot(X_train,regression.predict(X_train))


# In[44]:


plt.scatter(X_test,Y_test)
plt.plot(X_test,regression.predict(X_test))


# In[45]:


Y_pred=regression.predict(X_test)


# In[46]:


from sklearn.metrics import mean_absolute_error,mean_squared_error


# In[47]:


mse=mean_squared_error(Y_test,Y_pred)
mae=mean_absolute_error(Y_test,Y_pred)
rmse=np.sqrt(mse)
print(mse)
print(mae)
print(rmse)


# # R SQUARE
# Formula
# 
# **R^2 = 1 - SSR/SST**
# 
# 
# R^2 = coefficient of determination
# SSR = sum of squares of residuals
# SST = total sum of squares

# In[41]:


from sklearn.metrics import r2_score


# In[43]:


score=r2_score(Y_test,Y_pred)
print(score)


# # **Adjusted R2 = 1 - [(1-R2)*(n-1)/(n-k-1)]**
# 
# where:
#     
# R2: The R2 of the model
# n: The number of observation
# k: The number of prdictor variables
# 

# In[44]:


#display adjusted R-squared
1 - (1-score)*(len(Y_test)-1)/(len(Y_test)-X_test.shape[1]-1)


# In[46]:


##OLS Linear Regression
import statsmodels.api as sm


# In[47]:


model=sm.OLS(Y_train,Y_train).fit()


# In[48]:


prediction=model.predict(X_test)
print(prediction)


# In[49]:


print(model.summary())


# In[50]:


##prediction for new data
regression.predict([[62]])


# In[51]:


regression.predict(scaler.transform([[56]]))


# # MULTIPLE LINEAR REGRESSION

# In[1]:


import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm


# In[2]:


data = pd.read_csv(r"C:\Users\Yuktha\Desktop\boston17.csv")


# In[4]:


data.head()


# In[6]:


data.shape


# In[7]:


data.info()


# In[8]:


data.isnull().sum()


# In[9]:


data.describe()


# In[10]:


plt.figure(figsize = (12, 12))
heatmap = sns.heatmap(data.corr(), vmin = -1, vmax = 1, annot = True)
heatmap.set_title('Correlation Heatmap')


# # REGRESSION WITH SKLEARN

# In[11]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score


# In[12]:


X = data.drop('MEDV', axis=1)
y = data['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[13]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[18]:


model = LinearRegression()
model.fit(X_train_scaled, y_train)


# In[16]:


predictions = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, predictions)
print(f"MSE: {mse:.4f}")


# In[17]:


r2 = r2_score(y_test, predictions)
print(f"R-squared: {r2:.4f}")


# In[19]:


plt.scatter(y_test, predictions, label='Prediction')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted x Actual Values')
plt.legend()
plt.show()


# In[20]:


final = pd.DataFrame({'Values': y_test.values,'Predictions': predictions})
final.head()


# In[39]:


import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
pd.set_option('display.float_format', lambda x: '%.3f' % x)
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


# In[40]:


data.skew()


# In[41]:


data.kurtosis()


# # OUTLIERS ANALYSIS

# In[42]:


for column in df:
    plt.figure(figsize = (17, 1))
    sns.boxplot(data = df, x = column)


# In[45]:


data.columns


# In[46]:


feature_list = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
                'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']


# # IQR METHOD

# In[47]:


def IQR_method (df,n,features):
    """
    Takes a dataframe and returns an index list corresponding to the observations 
    containing more than n outliers according to the Tukey IQR method.
    """
    outlier_list = []
    
    for column in features:
                
        # 1st quartile (25%)
        Q1 = np.percentile(df[column], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[column],75)
        
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        # outlier step
        outlier_step = 1.5 * IQR
        outlier_list_column = df[(df[column] < Q1 - outlier_step) | (df[column] > Q3 + outlier_step )].index
        
        # appending the list of outliers 
        outlier_list.extend(outlier_list_column)
        
    # selecting observations containing more than x outliers
    outlier_list = Counter(outlier_list)        
    multiple_outliers = list( k for k, v in outlier_list.items() if v > n )
    
    # Calculate the number of records below and above lower and above bound value respectively
    df1 = df[df[column] < Q1 - outlier_step]
    df2 = df[df[column] > Q3 + outlier_step]
    
    print('Total number of outliers is:', df1.shape[0]+df2.shape[0])
    
    return multiple_outliers


# In[48]:


Outliers_IQR = IQR_method(df, 1, feature_list)


# In[49]:


df_out = df.drop(Outliers_IQR, axis = 0).reset_index(drop=True)
df_out.head()


# In[50]:


for column in df_out:
    plt.figure(figsize = (17, 1))
    sns.boxplot(data = df_out, x = column)


# # STANDARD DEVIATION METHOD

# In[51]:


def StDev_method (df,n,features):
    """
    Takes a dataframe df of features and returns an index list corresponding to the observations 
    containing more than n outliers according to the standard deviation method.
    """
    outlier_indices = []
    
    for column in features:
        # calculate the mean and standard deviation of the data frame
        data_mean = df[column].mean()
        data_std = df[column].std()
        
        # calculate the cutoff value
        cut_off = data_std * 3
        
        # Determining a list of indices of outliers for feature column        
        outlier_list_column = df[(df[column] < data_mean - cut_off) | (df[column] > data_mean + cut_off)].index
        # appending the found outlier indices for column to the list of outlier indices 
        outlier_indices.extend(outlier_list_column)
        
    # selecting observations containing more than x outliers
    outlier_indices = Counter(outlier_indices)        
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    # Calculate the number of records below and above lower and above bound value respectively
    df1 = df[df[column] > data_mean + cut_off]
    df2 = df[df[column] < data_mean - cut_off]
    print('Total number of outliers is:', df1.shape[0]+ df2.shape[0])
    
    return multiple_outliers


# In[52]:


Outliers_StDev = StDev_method(df,1,feature_list)


# In[53]:


df_out2 = df.drop(Outliers_StDev, axis = 0).reset_index(drop = True)
df_out2.head()


# In[54]:


for column in df_out2:
    plt.figure(figsize = (17, 1))
    sns.boxplot(data = df_out2, x = column)


# # Z CORE METHOD

# In[55]:


def z_score_method (df,n,features):
    """
    Takes a dataframe df of features and returns an index list corresponding to the observations 
    containing more than n outliers according to the z-score method.
    """
    outlier_list = []
    
    for column in features:
        # calculate the mean and standard deviation of the data frame
        data_mean = df[column].mean()
        data_std = df[column].std()
        threshold = 3
        
        z_score = abs( (df[column] - data_mean)/data_std )
        
        # Determining a list of indices of outliers for feature column        
        outlier_list_column =  df[z_score > threshold].index
        
        # appending the found outlier indices for column to the list of outlier indices 
        outlier_list.extend(outlier_list_column)
       # selecting observations containing more than x outliers
    outlier_list = Counter(outlier_list)        
    multiple_outliers = list( k for k, v in outlier_list.items() if v > n )
    
    # Calculate the number of outlier records
    df1 = df[z_score > threshold]
    print('Total number of outliers is:', df1.shape[0])
    
    return multiple_outliers


# In[56]:


Outliers_z_score = z_score_method(df, 1, feature_list)


# In[57]:


df_out3 = df.drop(Outliers_z_score, axis = 0).reset_index(drop = True)
df_out3.head()


# In[58]:


for column in df_out3:
    plt.figure(figsize = (17, 1))
    sns.boxplot(data = df_out3, x = column)


# # FINAL RESULT

# In[59]:


for column in df:
    plt.figure(figsize = (17, 1))
    sns.boxplot(data = df, x = column)


# # IQR REVISITED

# In[60]:


for col in df:
  q1 = df[col].quantile(0.25)
  q3 = df[col].quantile(0.75)
  iqr = q3 - q1
  whisker_width = 1.5
  lower_whisker = q1 - (whisker_width * iqr)
  upper_whisker = q3 + whisker_width * iqr
  df[col] = np.where(df[col] > upper_whisker, upper_whisker, np.where(df[col] < lower_whisker, lower_whisker, df[col]))


# In[61]:


for column in df:
    plt.figure(figsize = (17, 1))
    sns.boxplot(data = df, x = column)


# # LOGISTIC REGRESSION

# # BINARY CLASSIFICATION

# In[86]:


import seaborn as sns
import pandas as pd
import numpy as np


# In[87]:


df=sns.load_dataset("iris")
df.head()


# In[88]:


df['species'].unique()


# In[89]:


df.isnull().sum()


# In[90]:


df=df[df['species']!='setosa']


# In[91]:


df.head()


# In[94]:


df['species'].map({'versicolor':6,'virginica':2})


# In[95]:


df['species']=df['species'].map({'versicolor':6,'virginica':2})


# In[96]:


df.head()


# In[97]:


df.isnull().sum()


# # split dataset into independent and dependent features

# In[98]:


X=df.iloc[:,:-1]


# In[99]:


Y=df.iloc[:,-1]


# In[100]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(
X,Y,test_size=0.25,random_state=42)


# In[101]:


from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()


# In[102]:


from sklearn.model_selection import GridSearchCV
parameter={'penalty':['l1','l2','elasticnet'],'C':[1,5,40,50,36,34,20,49,6,26,56],'max_iter':[100,200,300]}


# In[103]:


classifier_regressor=GridSearchCV(classifier,param_grid=parameter,scoring='accuracy',cv=5)


# In[104]:


classifier_regressor.fit(X_train,Y_train)


# # BEST PARAMETERS SELECTED

# In[107]:


print(classifier_regressor.best_params_)


# # ACCURACY RATE

# In[108]:


print(classifier_regressor.best_score_)


# # PREDICTION

# In[112]:


classifier_regressor.predict(X_test)


# In[113]:


Y_pred=classifier_regressor.predict(X_test)


# # CALCULATING ACCURACY SCORE

# In[114]:


from sklearn.metrics import accuracy_score,classification_report


# In[115]:


score=accuracy_score(Y_pred,Y_test)


# In[116]:


print(score)


# In[117]:


classification_report(Y_pred,Y_test)


# In[118]:


print(classification_report(Y_pred,Y_test))


# # EDA

# In[119]:


sns.pairplot(df,hue='species')


# In[120]:


df.corr()


# # DECISION TREE

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.datasets import load_iris


# In[4]:


iris=load_iris()


# In[5]:


iris


# In[6]:


iris.data


# In[7]:


iris.target


# In[8]:


import seaborn as sns


# In[11]:


sns.load_dataset("iris")


# In[12]:


df=sns.load_dataset("iris")


# # INDEPENDENT AND DEPENDENT FEATURES

# In[14]:


X=df.iloc[:,:-1]
Y=iris.target


# In[15]:


X


# In[16]:


Y


# In[17]:


X,Y


# # TRAIN TEST SPLIT

# In[21]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(
X,Y,test_size=0.25,random_state=42)


# In[22]:


X_train


# In[23]:


from sklearn.tree import DecisionTreeClassifier


# # POST PRUNING

# In[26]:


treemodel=DecisionTreeClassifier()


# In[27]:


treemodel.fit(X_train,Y_train)


# In[28]:


from sklearn import tree
plt.figure(figsize=(16,10))
tree.plot_tree(treemodel,filled=True)


# # PREDICTION

# In[29]:


Y_pred=treemodel.predict(X_test)


# In[30]:


Y_pred


# In[31]:


from sklearn.metrics import accuracy_score,classification_report 


# In[32]:


score=accuracy_score(Y_pred,Y_test)
print(score)


# In[34]:


classification_report(Y_pred,Y_test)


# In[35]:


print(classification_report(Y_pred,Y_test))


# # PRE PRUNING 

# In[46]:


parameter={
    'criterion':['gini','entropy','log_loss'],
    'splitter':['best','random'],
    'max_depth':[1,2,3,4,5],
    'max_features':['auto','sqrt','log2']
}


# In[47]:


from sklearn.model_selection import GridSearchCV


# In[48]:


treemodel=DecisionTreeClassifier()
cv=GridSearchCV(treemodel,param_grid=parameter,cv=6,scoring='accuracy')


# In[49]:


cv.fit(X_train,Y_train)


# In[50]:


cv.best_params_


# In[51]:


cv.predict(X_train)


# In[55]:


Y_pred=cv.predict(X_test)


# In[56]:


from sklearn.metrics import accuracy_score,classification_report


# In[57]:


score=accuracy_score(Y_pred,Y_test)


# In[58]:


score


# In[59]:


classification_report(Y_pred,Y_test)


# In[60]:


print(classification_report(Y_pred,Y_test))


# # RANDOM FOREST 

# In[131]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


# In[132]:


titanic=pd.read_csv(r"C:\Users\Yuktha\Desktop\titanic.csv")
print(titanic.shape)


# In[133]:


titanic.head(6)


# In[134]:


NA=pd.concat([titanic.isnull().sum()], axis=1, keys=["Titanic"])
NA[NA.sum(axis=1) > 0]


# In[135]:


titanic["Age"] = titanic["Age"].fillna(titanic["Age"].mean())


# In[136]:


titanic["Cabin"] = titanic["Cabin"].fillna(titanic["Cabin"].mode())


# In[137]:


titanic["Embarked"] = titanic["Embarked"].fillna(titanic["Embarked"].mode())


# In[138]:


titanic["Pclass"] = titanic["Pclass"].apply(str)


# In[139]:


for col in titanic.dtypes[titanic.dtypes == "object"].index:
    for_dummy = titanic.pop(col)
    titanic = pd.concat([titanic,pd.get_dummies(for_dummy,prefix=col)],axis=1)
titanic.head()


# In[140]:


labels = titanic.pop("Survived")


# In[145]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(titanic, labels, test_size=0.25)


# In[146]:


from sklearn.ensemble import RandomForestClassifier


# In[147]:


rf = RandomForestClassifier()


# In[148]:


rf.fit(X_train,Y_train)


# In[149]:


Y_pred=rf.predict(X_test)


# In[151]:


from sklearn.metrics import roc_curve,auc
false_positive_rate,true_positive_rate,thresholds=roc_curve(Y_test,Y_pred)
roc_auc=auc(false_positive_rate,true_positive_rate)
roc_auc


# In[155]:


n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]
train_results = []
test_results = []


# In[160]:


for estimator in n_estimators:
    rf = RandomForestClassifier(n_estimators=estimator, n_jobs=-1)
    rf.fit(X_train,Y_train)
    train_pred = rf.predict(X_train)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_train,train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    Y_pred = rf.predict(X_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test,Y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)


# In[161]:


from matplotlib.legend_handler import HandlerLine2D
line1, = plt.plot(n_estimators, train_results, "b", label="Train AUC")
line2,=plt.plot(n_estimators,test_results,"r",label="Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel("AUC score")
plt.xlabel("n_estimators")
plt.show()


# # SUPPORT VECTOR MACHINE 

# In[4]:


import numpy as np
import matplotlib.pyplot as plt


# In[5]:


x=np.linspace(-5.0,5.0,100)
y=np.sqrt(10**2 - x**2)
y=np.hstack([y,-y])
x=np.hstack([x,-x])


# In[6]:


x1=np.linspace(-5.0,5.0,100)
y1=np.sqrt(5**2 - x1**2)
y1=np.hstack([y1,-y1])
x1=np.hstack([x1,-x1])


# In[7]:


plt.scatter(y,x)
plt.scatter(y1,x1)


# In[8]:


import pandas as pd
df1=pd.DataFrame(np.vstack([y,x]).T,columns=['X1','X2'])
df1['Y']=0
df2=pd.DataFrame(np.vstack([y1,x1]).T,columns=['X1','X2'])
df2['Y']=1
df=df1.append(df2)
df.head(5)


# In[9]:


x=df.iloc[:,:2]
y=df.Y


# In[10]:


y


# In[11]:


x


# In[14]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# In[15]:


Y_train


# In[16]:


X_train


# In[17]:


X_test


# In[18]:


Y_test


# In[19]:


from sklearn.svm import SVC
classifier=SVC(kernel="linear")
classifier.fit(X_train,Y_train)


# In[20]:


from sklearn.metrics import accuracy_score
Y_pred=classifier.predict(X_test)
accuracy_score(Y_test,Y_pred)


# In[21]:


df.head()


# # POLYNOMIAL KERNEL  --  K(x,y)=(x^Ty+c)^d

# In[22]:


#Finding components for polynomial kernel
df['X1_Square']=df['X1']**2
df['X2_Square']=df['X2']**2
df['X1*X2']=(df['X1']*df['X2'])
df.head()


# In[23]:


X=df[['X1','X2','X1_Square','X2_Square','X1*X2']]
y=df['Y']


# In[24]:


y


# In[25]:


X_train, X_test, y_train, y_test = train_test_split(X,y,
                                               test_size=0.25,
                                               random_state=0)


# In[26]:


X_train


# In[27]:


import plotly.express as px
fig=px.scatter_3d(df,x='X1',y='X2',z='X1*X2',color='Y')
fig.show()


# In[28]:


classifier=SVC(kernel="linear")
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
accuracy_score=(y_test,y_pred)


# In[58]:


print(accuracy_score)


# # XAI MODEL

# In[18]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[19]:


dataset = pd.read_csv(r"C:\Users\Yuktha\Desktop\Churn_modeling.csv")
X=dataset.iloc[:,3:13]
Y=dataset.iloc[:,13]


# In[20]:


dataset.head()


# In[21]:


dataset.isnull().sum()


# In[22]:


geography=pd.get_dummies(X["Geography"],drop_first=True)
gender=pd.get_dummies(X["Gender"],drop_first=True)


# In[23]:


X=pd.concat([X,geography,gender],axis=1)


# In[24]:


X=X.drop(['Geography','Gender'],axis=1)


# In[25]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)


# In[28]:


from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()
classifier.fit(X_train,Y_train)


# In[29]:


import pickle
pickle.dump(classifier,open("classifier.pki","wb"))


# In[33]:


pip install lime


# In[42]:


import lime
from lime import lime_tabular
interpretor=lime_tabular.LimeTabularExplainer(
training_data=np.array(X_train),
    feature_names=X_train.columns,
    mode="classification"
)


# In[43]:


X_test.iloc[4]


# In[44]:


exp=interpretor.explain_instance(
    data_row=X_test.iloc[4],
    predict_fn=classifier.predict_proba
)

exp.show_in_notebook(show_table=True)


# # Shapash - Xai

# In[32]:


import seaborn as sns


# In[33]:


df=sns.load_dataset("tips")


# In[34]:


df.head()


# In[35]:


Y=df["tip"]
X=df[df.columns.difference(["tip"])]


# In[36]:


X.head()


# In[37]:


df.info()


# In[38]:


X["day"]=X["day"].cat.codes
X["sex"]=X["sex"].cat.codes
X["smoker"]=X["smoker"].cat.codes
X["time"]=X["time"].cat.codes


# In[39]:


X


# In[40]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.75,random_state=42)


# In[41]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=200).fit(X_train,Y_train)


# In[42]:


get_ipython().system('pip install shapash')


# In[48]:


from shapash.explainer.smart_explainer import SmartExplainer


# In[73]:


xpl = SmartExplainer(regressor)


# In[75]:


xpl.compile(
    x=X_test,
)


# In[78]:


xpl


# In[77]:


app=xpl.run_app(title_story="Tips Dataset")


# In[79]:


predictor=xpl.to_smartpredictor()


# In[80]:


predictor.save("./predictor.pkl")


# In[82]:


from shapash.utils.load_smartpredictor import load_smartpredictor
predictor_load=load_smartpredictor("./predictor.pkl")


# In[86]:


predictor_load.add_input(x=X, ypred=Y)


# In[88]:


detailed_contributions=predictor_load.detail_contributions()


# In[89]:


detailed_contributions.head()


# In[94]:



predictor_load.modify_mask(max_contrib=3)


# # TEXT CLASSIFICATION

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.datasets import fetch_20newsgroups
data=fetch_20newsgroups()
data.target_names


# In[4]:


#defining all categories
categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']


# In[5]:


#training the data on these categories
train=fetch_20newsgroups(subset='train',
categories=categories)


# In[6]:


#testing the data for these categories
test=fetch_20newsgroups(subset='test',
categories=categories)


# In[7]:


#printing the train data
print(train.data)


# In[8]:


#printing the test data
print(test.data)


# In[10]:


print(train.data[2])


# In[11]:


print(len(train.data))


# In[13]:


#importing necessary packages
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


# In[14]:


#creating a model based on multinomial Naive Bayes
model=make_pipeline(TfidfVectorizer(), MultinomialNB())


# In[15]:


#training the model with the train data 
model.fit(train.data, train.target)


# In[17]:


#creating labels for the test data
labels=model.predict(test.data)


# In[20]:


#creating confusion matrix and heat map
from sklearn.metrics import confusion_matrix
mat=confusion_matrix(test.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d',
cbar=False, xticklabels=train.target_names,
yticklabels=train.target_names)

#plotting heatmap of confusion matrix
plt.xlabel('true label')
plt.ylabel('predicted label')


# In[21]:


#predicting category on new data based on trained model
def predict_category(s, train=train, model=model):
    pred=model.predict([s])
    return train.target_names[pred[0]]


# In[22]:


predict_category ('cycles')


# In[23]:


predict_category('Sending load to International Space Station')


# In[24]:


predict_category('BMW is better than benz')


# # Embeddings - Word2Vec

# In[3]:


file=open(r"C:\Users\Yuktha\Desktop\Internship details\yuktha_data.txt")
yuktha_data=file.readlines()
file.close()
print(yuktha_data)


# In[4]:


for i in range(len(yuktha_data)):
    yuktha_data[i]=yuktha_data[i].lower().replace('\n', '')
print(yuktha_data)


# # Remove stopwords and tokenize 

# In[6]:


stopwords=['the','is','are','can','will','be','a','only','their','now','and','at','it']
filtered_data=[]
for sent in yuktha_data:
    temp=[]
    for word in sent.split():
        if word not in stopwords:
            temp.append(word)
    filtered_data.append(temp)
print(filtered_data)


# # Creating Bigrams

# In[9]:


bigrams=[]
for words_list in filtered_data:
    for i in range(len(words_list)-1):
        for j in range(i+1, len(words_list)):
            bigrams.append([words_list[i], words_list[j]])
            bigrams.append([words_list[j], words_list[i]])
print(bigrams)


# # Vocabulary

# In[10]:


all_words=[]
for sent in filtered_data:
    all_words.extend(sent)
all_words=list(set(all_words))
all_words.sort()
print(all_words)
print(len(all_words))


# # One-hot encoding

# In[12]:


words_dict={}
counter=0
for word in all_words:
    words_dict[word]=counter
    counter +=1
print(words_dict)


# In[15]:


import numpy as np
onehot_data=np.zeros((len(all_words), len(all_words)))
for i in range(len(all_words)):
    onehot_data[i][i]=1
print(onehot_data)
onehot_dict={}
for i in range(len(all_words)):
    onehot_dict[all_words[i]]=onehot_data[i]
for word in onehot_dict:
    print(word, ":", onehot_dict[word])   


# # Principal Component Analysis(PCA):

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from sklearn.datasets import load_breast_cancer


# In[4]:


cancer=load_breast_cancer()


# In[5]:


cancer.keys()


# In[6]:


print(cancer['DESCR'])


# In[7]:


df=pd.DataFrame(cancer['data'],columns=cancer['feature_names'])


# In[8]:


df.head()


# In[9]:


from sklearn.preprocessing import MinMaxScaler


# In[11]:


from sklearn.preprocessing import StandardScaler


# In[12]:


scaler=StandardScaler()
scaler.fit(df)


# In[13]:


scaled_data=scaler.transform(df)


# In[14]:


scaled_data


# In[15]:


from sklearn.decomposition import PCA


# In[20]:


pca=PCA(n_components=2)


# In[21]:


pca.fit(scaled_data)


# In[22]:


x_pca=pca.transform(scaled_data)


# In[23]:


scaled_data.shape


# In[24]:


x_pca.shape


# In[25]:


scaled_data


# In[26]:


x_pca


# In[27]:


plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'])
plt.xlabel('First principle component')
plt.ylabel('Second principle component')


# # Transformer and BERT implementation

# In[1]:


get_ipython().system('pip install transformers')


# In[2]:


from transformers import pipeline 
classifier=pipeline('sentiment-analysis')


# In[5]:


classifier('We are very happy to show you the 🤗 Transformers library')


# In[6]:


classifier('The food is not so great')


# In[9]:


results=classifier(["We are very happy to show you the 🤗 Transformers library", "We hope you don't hate it"])
for result in results:
    print(f"label: {result['label']}, with score: {round(result['score'],4)}")


# # XLNET

# In[1]:


get_ipython().system('pip install -q -U watermark')


# In[6]:


get_ipython().system('pip install torch')


# In[7]:


get_ipython().run_line_magic('reload_ext', 'watermark')
get_ipython().run_line_magic('watermark', '-v -p numpy,pandas,torch,transformers')


# In[23]:


import transformers
from transformers import XLNetTokenizer,XLNetModel,AdamW,get_linear_schedule_with_warmup
import torch


# In[24]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from collections import defaultdict
from textwrap import wrap
from pylab import rcParams


# In[25]:


from torch import nn,optim
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset,RandomSampler,SequentialSampler
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


# In[26]:


device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device


# In[27]:


df=pd.read_csv(r"C:\Users\Yuktha\Desktop\Internship details\IMDB Dataset.csv")


# In[28]:


df.head()


# In[29]:


from sklearn.utils import shuffle
df=shuffle(df)
df.head()


# In[30]:


df=df[:24000]
len(df)


# In[31]:


import re
def clean_text(text):
    text=re.sub(r"@[A-Za-z0-9]+", '', text)
    text=re.sub(r"https?://[A-Za-z0-9./]+",'',text)
    text=re.sub(r"[^a-zA-z.!?'0-9]", '', text)
    text=re.sub('/t','',text)
    text=re.sub(r" +", '', text)
    return text


# In[32]:


df['review']=df['review'].apply(clean_text)


# In[33]:


rcParams['figure.figsize']=8,6
sns.countplot(df.sentiment)
plt.xlabel('review score')


# In[34]:


def sentiment2label(sentiment):
    if sentiment=="positive":
        return 1
    else:
        return 0
df['sentiment']=df['sentiment'].apply(sentiment2label)


# In[35]:


df['sentiment'].value_counts()


# In[36]:


class_names=['negative','positive']


# # XLNET TOKENIZER

# In[37]:


get_ipython().system('pip install SentencePiece')


# In[38]:


pip install sentencepiece


# In[39]:


from transformers import XLNetTokenizer,XLNetModel
PRE_TRAINED_MODEL_NAME = 'xlnet-base-cased'
tokenizer=XLNetTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)


# In[40]:


input_txt="India is my country. All Indians are my brothers and sisters"
encodings=tokenizer.encode_plus(input_txt,add_special_tokens=True, max_length=16, return_tensors='pt', return_tokens=tokenizer.encode("India is my country. All Indians are my brothers and sisters"), truncation=True)


# In[41]:


print('input_ids : ',encodings['input_ids'])


# In[42]:


tokenizer.convert_ids_to_tokens(encodings['input_ids'][0])


# In[43]:


type(encodings['attention_mask'])


# In[45]:


attention_mask=pad_sequences(encodings['attention_mask'],maxlen=512,dtype=torch.Tensor,truncating="post",padding="post")


# In[47]:


attention_mask=attention_mask.astype(dtype='int64')
attention_mask=torch.tensor(attention_mask)
attention_mask.flatten()


# In[48]:


encodings['input_ids']


# In[55]:


token_lens=[]
for txt in df['review']:
    tokens=tokenizer.encode(txt,max_length=512)
    token_lens.append(len(tokens))


# In[58]:


sns.distplot(token_lens)
plt.xlim([0,1024]);
plt.xlabel('Token count');


# In[59]:


MAX_LEN=512


# # ERNIE

# In[3]:


import numpy as np
import pandas as pd


# In[4]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[5]:


get_ipython().system('pip install ernie')


# In[6]:


from ernie import SentenceClassifier, Models 
import pandas as pd


# In[7]:


df=pd.read_csv(r"C:\Users\Yuktha\Desktop\Internship details\IMDB Dataset.csv")
df.head()


# In[8]:


df['sentiment'].map({'positive':6, 'negative':2})


# In[9]:


train=pd.read_csv(r"C:\Users\Yuktha\Desktop\Internship details\IMDB Dataset.csv", usecols=['review','sentiment'])
test=pd.read_csv(r"C:\Users\Yuktha\Desktop\Internship details\IMDB Dataset.csv")
sub_sample=pd.read_csv(r"C:\Users\Yuktha\Desktop\Internship details\IMDB Dataset.csv")
print(train.shape,test.shape,sub_sample.shape)


# In[10]:


train=train.rename(columns={'review':0,'sentiment':1})
train.head()


# In[11]:


import seaborn as sns
sns.countplot(y=train[1])


# In[12]:


classifier=SentenceClassifier(model_name=Models.BertBaseUncased, max_length=128, labels_no=2)


# In[15]:


from ernie import SplitStrategies, AggregationStrategies
text="ablaze,London,Birmingham Wholesale Market is ablaze BBC News -Fire breaks out at Birmingham's Wholesale Market http://t.co/irWqCEZWEU"
probabilities=classifier.predict_one(text, aggregation_strategy=AggregationStrategies.Mean)


# In[16]:


probabilities


# # TEXT-TO-TEXT-TRANSFER-TRANSFORMER

# In[2]:


get_ipython().system('pip install datasets transformers rouge-score nltk')


# In[3]:


import transformers
print(transformers.__version__)


# In[4]:


model_checkpoint="t5-small" 


# # Deep Learning

# In[2]:


import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv(r"C:\Users\Yuktha\Desktop\Internship details\Telecom customer churn.csv")
df.sample(5)


# In[4]:


df.drop('customerID',axis='columns',inplace=True)
df.dtypes


# In[5]:


df.TotalCharges.values


# In[6]:


df.MonthlyCharges.values


# In[7]:


pd.to_numeric(df.TotalCharges,errors='coerce')


# In[8]:


pd.to_numeric(df.TotalCharges,errors='coerce').isnull()


# In[9]:


df[pd.to_numeric(df.TotalCharges,errors='coerce').isnull()]


# In[10]:


df.shape


# In[11]:


df.iloc[488]


# In[12]:


df.iloc[488]['TotalCharges']


# In[13]:


df1 = df[df.TotalCharges!=' ']
df1.shape


# In[14]:


df1.dtypes


# In[15]:


pd.to_numeric(df1.TotalCharges)


# In[16]:


df1.TotalCharges = pd.to_numeric(df1.TotalCharges)


# In[17]:


df1.TotalCharges.dtypes


# In[18]:


df1[df1.Churn=='No']


# In[19]:


df1[df1.Churn=='No'].tenure


# In[20]:


tenure_churn_no = df1[df1.Churn=='No'].tenure
tenure_churn_yes = df1[df1.Churn=='Yes'].tenure

plt.xlabel("tenure")
plt.ylabel("Number of Customers")
plt.title("Customer Churn Prediction Visualization")

plt.hist([tenure_churn_yes, tenure_churn_no], color = ['blue', 'orange'], label = ['ChurnYes', 'ChurnNo'])
plt.legend()


# In[21]:


mc_churn_no = df1[df1.Churn=='No'].MonthlyCharges
mc_churn_yes = df1[df1.Churn=='Yes'].MonthlyCharges

plt.xlabel("Monthly Charges")
plt.ylabel("Number of Customers")
plt.title("Customer Churn Prediction Visualization")

blood_sugar_men = [113, 85, 90, 50, 149, 88, 93, 115, 135, 80, 77, 82, 129]
blood_sugar_women = [67, 98, 89, 120, 133, 150, 84, 69, 89, 79, 120, 112, 100]

plt.hist([mc_churn_yes, mc_churn_no], rwidth = 0.95, color = ['green', 'red'], label = ['Churn=Yes', 'Churn=No'])
plt.legend()


# In[22]:


for column in df:
    print(column)


# In[23]:


for column in df:
    print(df[column].unique())


# In[24]:


for column in df:
    print(df[column].unique())


# In[26]:


def print_unique_col_values(df):
    for column in df:
        if df[column].dtypes=='object':
            print(f'{column} : {df[column].unique()}')


# In[27]:


print_unique_col_values(df1)


# In[28]:


df1.replace('No internet service', 'No', inplace=True)
df1.replace('No phone service', 'No', inplace=True)


# In[29]:


print_unique_col_values(df1)


# In[30]:


yes_no_columns = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                  'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Churn']

for col in yes_no_columns:
    df1[col].replace({'Yes':1, 'No':0}, inplace=True)


# In[31]:


for col in df1:
    print(f'{col}: {df1[col].unique()}')


# In[32]:


df1['gender'].replace({'Female':1, 'Male':0}, inplace=True)


# In[33]:


df1['gender'].unique()


# In[35]:


pd.get_dummies(data=df1, columns=['InternetService'])


# In[36]:


df2 = pd.get_dummies(data=df1, columns=['InternetService', 'Contract', 'PaymentMethod'])
df2.columns


# In[37]:


df2.sample(5)


# In[38]:


df2.dtypes


# In[39]:


cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df2[cols_to_scale]=scaler.fit_transform(df2[cols_to_scale])


# In[40]:


df2.sample(4)


# In[41]:


for col in df2:
    print(f'{col}: {df2[col].unique()}')


# In[49]:


X = df2.drop('Churn', axis='columns')
Y = df2['Churn']


# In[50]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=5)


# In[51]:


X_train.shape


# In[52]:


X_test.shape


# In[53]:


X_train[:10]


# In[54]:


len(X_train.columns)


# In[56]:


import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Dense(20, input_shape=(26,), activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=100)


# In[58]:


model.evaluate(X_test, Y_test)


# In[59]:


yp = model.predict(X_test)
yp[:6]


# In[60]:


Y_test[:6]


# In[61]:


Y_pred = []
for element in yp:
    if element > 0.5:
        Y_pred.append(1)
    else:
        Y_pred.append(0)


# In[62]:


Y_pred[:6]


# In[63]:


from sklearn.metrics import confusion_matrix, classification_report

print(classification_report(Y_test, Y_pred))


# In[64]:


import seaborn as sn
cm = tf.math.confusion_matrix(labels = Y_test, predictions = Y_pred)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt = 'd')
plt.xlabel('Predicted')
plt.ylabel('Truth')


# In[65]:


Y_test.shape


# # ACCURACY 

# In[68]:


round((862+229)/(862+229+137+179), 2)


# # Precision for 0 class i.e., Precision for customers who did not churn

# In[70]:


round(862/(862+179),2)


# # Precision for 1 class i.e., Precision for customers who actually churned

# In[71]:


round(229/(229+137), 2)


# # Recall for 0 class

# In[72]:


round(862/(862+137), 2)


# In[73]:


round(229/(229+179), 2)


# In[ ]:
