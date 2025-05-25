import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("LoanApprovalPrediction.csv")
datac = data.copy()

def loan_status(status):
    if status == 'Y':
        return 'Approved'
    elif status == 'N':
        return 'rejected'
    else:
        return 'Unknown'

datac['Loan_Status'] = datac['Loan_Status'].apply(loan_status)
yes_df = datac[datac['Loan_Status'] == 'Approved'].reset_index(drop=True)
yes_df['LoanAmount']=yes_df['LoanAmount'].fillna(int(yes_df['LoanAmount'].mean()))
no_df = datac[datac['Loan_Status'] == 'rejected'].reset_index(drop=True)

print('----------x----------')
print(data.head(5))
obj = (data.dtypes == 'object')
print('----------x----------')
print("Categorical variables:",len(list(obj[obj].index)))

data.drop(['Loan_ID'],axis=1,inplace=True)

obj = (data.dtypes == 'object')
object_cols = list(obj[obj].index)
plt.figure(figsize=(18,36))

index = 1
for col in object_cols:
  y = data[col].value_counts()
  plt.subplot(2,3,index)
  plt.xticks(rotation=10)
  plt.title(f'{col} counts')
  sns.barplot(x=list(y.index), y=y)
  index +=1
# plt.tight_layout()
plt.show()

selected_col = ['Gender','Married','Education','Property_Area']
for scol in selected_col:
  #  x = yes_df[scol].value_counts()
   plt.subplot(1,2,1)
   plt.title("approved")
   sns.countplot(x=scol, data=yes_df)
   
  #  y = no_df[scol].value_counts()
   plt.subplot(1,2,2)
   plt.title("rejected")
   sns.countplot(x=scol, data=no_df)
   plt.show()


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
obj = (data.dtypes == 'object')
for col in list(obj[obj].index):
  data[col] = label_encoder.fit_transform(data[col])

obj = (data.dtypes == 'object')
print('----------x----------')
print("Categorical variables:",len(list(obj[obj].index)))

plt.figure(figsize=(12,6))
sns.heatmap(data.corr(),linewidths=2,annot=True)
plt.show()

for col in data.columns:
  data[col] = data[col].fillna(data[col].mean()) 
print('----------x----------')
print(data.isna().sum())

from sklearn.model_selection import train_test_split

X = data.drop(['Loan_Status'],axis=1)
Y = data['Loan_Status']
print('----------x----------')
print(X.shape,Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.2,random_state=42)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

knn = KNeighborsClassifier(n_neighbors=3)
rfc = RandomForestClassifier(n_estimators = 7,random_state =42)
svc = SVC()
lc = LogisticRegression()

# making predictions on the training set
print('----------x----------')
for clf in (rfc, knn, svc,lc):
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_train)
    print("Accuracy score of ",clf,"=",100*accuracy_score(Y_train, Y_pred))

# making predictions on the testing set
print('----------x----------')
for clf in (rfc, knn, svc,lc):
    clf.fit(X_train, Y_train)
    Y_pred = clf.predict(X_test)
    print("Accuracy score of ",clf,"=",100*accuracy_score(Y_test,Y_pred))



from sqlalchemy import create_engine

host = 'localhost'
user = 'root'
password = 'GUHAN23'
database = 'loan'


engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database}")
yes_df.to_sql(name='approved_loans', con=engine, if_exists='append', index=False)
print("Approved_loans Data inserted successfully")
no_df.to_sql(name='rejected_loans', con=engine, if_exists='append', index=False)
print("Rejected_loans Data inserted successfully")

import pymysql as mysql

con = mysql.connect(host='localhost',user='root',password='GUHAN23',database='loan')
cursor=con.cursor()

cursor.execute('select Loan_ID,LoanAmount,Loan_Status from approved_loans;')
print('----------x----------')
print("approved loan ids")
print(pd.DataFrame(cursor.fetchall(),columns=['Loan_ID','LoanAmount','Loan_Status']).to_string())

cursor.execute('select Loan_ID,LoanAmount,Loan_Status from rejected_loans;')
print('----------x----------')
print("rejected loan ids")
print(pd.DataFrame(cursor.fetchall(),columns=['Loan_ID','LoanAmount','Loan_Status']).to_string())