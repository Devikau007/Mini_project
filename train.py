import pandas as pd 
import numpy as np  
import pickle  
import sklearn.ensemble as ske 
from sklearn import cross_validation 
from sklearn.feature_selection import SelectFromModel
from sklearn.externals import joblib
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
data = pd.read_csv('data.csv', sep='|')
X = data.drop(['Name', 'md5', 'legitimate'], axis=1).values  #to drop unnecessary colmns
y = data['legitimate'].values

print('Finding Important Features from %i total features\n' % X.shape[1]) #total number of attributes

# Feature selection using Trees Classifier
fetrs = ske.ExtraTreesClassifier().fit(X, y) #build the forest of trees
model = SelectFromModel(fetrs, prefit=True) #Meta-transformer for selecting features based on importance weights.
X_new = model.transform(X) #Reduce X to the selected features.
no_feat= X_new.shape[1]

#Split arrays or matrices into random train and test subsets
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_new, y ,test_size=0.2)  

features = []

print('%i Important Features are Found :' % no_feat)

indices = np.argsort(fetrs.feature_importances_)[::-1][:no_feat]  
for f in range(no_feat):
    print("%d. feature %s (%f)" % (f + 1, data.columns[2+indices[f]], fetrs.feature_importances_[indices[f]]))

#take care of the feature order
for f in sorted(np.argsort(fetrs.feature_importances_)[::-1][:no_feat]):
    features.append(data.columns[2+f])

print("\nTesting SVM algorithm")
algo="SVM"

#clf = svm.SVC(gamma=0.009, C=100)
#clf = svm.SVC(gamma=0.001, C=100)
clf = svm.SVC(gamma=5, C=10)

clf.fit(X_train, y_train)
class_result = clf.score(X_test, y_test)
print("%s : %f %%" % (algo, class_result*100))
result = class_result
winner=algo
print('\nPrediction of SVM  algorithm is %f %% success' % (result*100))

# Save the algorithm and the feature list for later predictions
print('Saving algorithm and feature list in classifier directory...')
joblib.dump(clf, 'classifier/classifier.pkl')  #Persist an arbitrary Python object into one file
open('classifier/features.pkl', 'w').write(pickle.dumps(features))
print('\nFeatures are Saved. Proceed to check the file... ')


res = clf.predict(X_test)
mt = confusion_matrix(y_test, res)
print("False positive rate : %f %%" % ((mt[0][1] / float(sum(mt[0])))*100))
print('False negative rate : %f %%' % ( (mt[1][0] / float(sum(mt[1]))*100)))


