#import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import r2_score
#import the dataset
df=pd.read_csv(r"C:\datascience\dataset\StudentPerformanceFactors.csv")
print(df)
#Check whether there is any missing values
print(df.isna().sum())
df=df.dropna()
print(df)
print(df.describe)
print(df.info)

# Compute correlation matrix
correlation = df.corr()

#Print correlation with Exam_Score
print("📌 Correlation with Exam_Score:")
print(correlation['Exam_Score'].sort_values(ascending=False))

#Plot heatmap
plt.figure(figsize=(12,8))
sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


#2 DATA PREPROCESSING
# Seperate features and target
x=df.drop('Exam_Score',axis=1)
y=df['Exam_Score']


#Encode categorical variables
categorical_cols=x.select_dtypes(include='object').columns
x=pd.get_dummies(x,columns=categorical_cols,drop_first=True)

# Feature scaling
scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)

#3 Train-Test Split
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.2,random_state=42)

#4 MODEL BUILDING
'''lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred_lr=lr.predict(x_test)'''

#randomforest
rf=RandomForestRegressor()
rf.fit(x_train,y_train)
y_pred_rf=rf.predict(x_test)

#5 evaluation
print("Mean Absolute Error:",metrics.mean_absolute_error(y_test,y_pred_rf))
print('Mean squared Error:',metrics.mean_squared_error(y_test,y_pred_rf))
print("Root Mean squared Error:",np.sqrt(metrics.mean_squared_error(y_test,y_pred_rf)))

#check accuracy 

score=r2_score(y_test,y_pred_rf)
print("r2 score is:",score*100,"%")

'''print(" Accuracy:",accuracy_score(y_test,y_pred_rf))'''
