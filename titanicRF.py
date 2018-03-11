import pandas as pd
import numpy as np
import random as rnd

# Data Visualization
import seaborn as sns
from IPython.display import display
import matplotlib.pyplot as plt

# Machine Learning

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

global train_df
global test_df
global combine


def visualizeNumericalCorrelationA( data, feature1, feature2 ):
 g = sns.FacetGrid(data, col=feature2)
 g.map(plt.hist, feature1, bins=20)
 g.savefig("outputA.png")


def visualizeNumericalCorrelationF( data, feature1, feature2 ):
 g = sns.FacetGrid(data, col=feature2)
 g.map(plt.hist, feature1, bins=20)
 g.savefig("outputF.png")

def visualizeNumericalCorrelationS( data, feature1, feature2 ):
 g = sns.FacetGrid(data, col=feature2)
 g.map(plt.hist, feature1, bins=20)
 g.savefig("outputS.png")
# Training and Testing Data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = np.concatenate((train_df,test_df),axis=0)
combine=combine[~np.isnan(combine).any(axis=1)]

visualizeNumericalCorrelationF(train_df,'Fare','Survived')
visualizeNumericalCorrelationA(train_df,'Age','Survived')
visualizeNumericalCorrelationS(train_df,'Sex','Survived')

X=combine[:,[0,1,2,3]]
y=combine[:,4]

#shuffling and spliting
from sklearn.utils import shuffle
X, y = shuffle(X, y, random_state=0)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)


rf = RandomForestClassifier(n_estimators=100)

rf.fit(X_train,y_train)
pred=rf.predict(X_test)


pc=input('Enter the pclass:')
sex=input('Enter the gender:')
age=input('Enter the age:')
fare=input('Enter the fare:')


xm=[pc,sex,age,fare]
xm=np.array(xm)
xm=xm.reshape(1,-1)
ym=rf.predict(xm)

if ym==1:
	print('\nSurvived')
else:
	print('\nnot survived')

print('\n\n')
print('----Performance Analysis----')
print('\n')
print('Training Accuracy=',rf.score(X_train,y_train))
print('\n')
print('Testing Accuracy=',rf.score(X_test,y_test))
print('\n')



from sklearn.metrics import classification_report,confusion_matrix
print('Confusion Matrix:')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))



import pylab as pl
from sklearn.metrics import roc_curve, auc
y_roc = np.array(y_test)
fpr, tpr, thresholds = roc_curve(y_roc, pred)
roc_auc = auc(fpr, tpr)
print("\n Area under the ROC curve : %f" % roc_auc)
pl.clf()
pl.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
pl.plot([0, 1], [0, 1], 'k--')
pl.xlim([0.0, 1.0])
pl.ylim([0.0, 1.0])
pl.xlabel('False Positive Rate')
pl.ylabel('True Positive Rate')
pl.legend(loc="lower right")
pl.show()




