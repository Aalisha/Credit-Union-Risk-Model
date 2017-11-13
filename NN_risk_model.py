import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler  
import sklearn.metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import sknn
from sknn.mlp import Layer
from sknn.mlp import Classifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

file_test="C:\Users\...\...\cs-test_cleaned.csv";
file_train="C:\Users\...\...\cs-training_cleaned.csv";
data=pd.read_csv(file_train);
train=data.drop('SeriousDlqin2yrs',axis=1);
target=data['SeriousDlqin2yrs']

cor=train.corr().abs();
s=cor.unstack();
so=s.order(kind="quicksort");

#sss[train_index]
X=np.array(train);
y=np.array(target);
data1=pd.read_csv(file_test);
test=data1.drop('SeriousDlqin2yrs',axis=1);
targetT=data1['SeriousDlqin2yrs'];
X_test=np.array(test);
y_test=np.array(targetT);
w_train = np.empty((y.shape[0],))

# Risky customers are much lesser in number
w_train[y == 0] = 0.8;
w_train[y == 1] = 2.1;

layers=[Layer(type='Tanh',units=5,dropout=0.25),Layer(type='Softmax')];
nn3 = Classifier(layers,valid_set=None,valid_size=0,batch_size=50, n_iter=50,learning_rule='sgd',learning_rate=0.05,verbose=True);

scaler = StandardScaler()  
scaler.fit(X)  
X_train = scaler.transform(X)  
# apply same transformation to test data
X_test = scaler.transform(X_test)  
#pipeline = Pipeline([('min/max scaler', MinMaxScaler(feature_range=(0.0, 1.0))),('neural network', nn2)])
start=time.time()
nn3.fit(X_train,y,w_train)
print time.time()-start  #31.698s

nn3.score(X_train,y,w_train)  #Score: 0.875
nn3.score(X_test,y_test)       #Score: 0.9243

y_test_pred=nn3.predict(X_test)
print classification_report(y_test,y_test_pred)
'''
 precision    recall  f1-score   support

          0       0.96      0.96      0.96     41925
          1       0.44      0.39      0.41      3075

avg / total       0.92      0.92      0.92     45000
'''

y_test_proba=nn3.predict_proba(X_test)
list_proba=[]
for i in range(y_test_proba.shape[0]):
    list_proba.append(y_test_proba[i][1])
list_proba=np.array(list_proba)

fpr,tpr, thresholds=roc_curve(y_test,list_proba);
ks=max(abs(tpr-fpr));
print ks #0.565

lift = []
for i in range(0,10):
    lift.append(tpr[thresholds >= np.percentile(list_proba,10*i)][-1])
lift.reverse()

lift1=lift;
for i in range(0,10):
    lift1[i]=lift1[i]/(0.1*(i+1))
print(lift1)
lift1=np.around(lift1,2);
    
import seaborn as sns
%matplotlib inline
sns.set_context("talk")
sns.set_style("whitegrid")
ax=sns.pointplot(x = range(10,110,10), y = lift1[range(0,10)])
[ax.text(p[0],p[1]+0.1, p[1], color='g') for p in zip(ax.get_xticks(),lift1[range(0,10)])]
ax.set(xlabel='Population', ylabel='Lift');

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='RF + LR')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

auc= roc_auc_score(y_test, list_proba, average='macro', sample_weight=None)




