# woe
woe is a Python library designed to convert categorical and continuous variables into weight of evidence. Weight of evidence is a statistical technique used in information theory to measure the "strength" of a relationship between a binary target variable and a predictor variable. The library can be used for data preprocessing in predictive modeling or machine learning projects.

## installation
```
pip3 install woe-conversion
```

## usage
```
from woe_conversion.woe import *

# create an instance of the WoeConversion class
woemodel = WoeConversion(binarytarget='survived', features=['categorical_variable_1','categorical_variable_2','continuous_variable_3'])

# fit the model using training data
woemodel.fit(train)

# transform the training and test data using the fitted model
transformedtrain = woemodel.transform(train)
transformedtest = woemodel.transform(test)
```

In the above code, WoeConversion is the class that is used to perform the conversion of variables to weight of evidence. The binarytarget parameter is the name of the binary target column in your dataset, and features is a list of the columns that you want to convert to weight of evidence.

Once the model has been created, it is fit to the training data using the fit method. This method calculates the weight of evidence for each category in the specified columns, and stores the results in the model object.

The transform method is then used to transform the training and test data using the fitted model. This method replaces the original columns with their weight of evidence equivalents.

## Note on Missing Values
The weight of evidence is also calculated for missing values. Therefore, missing values should not be imputed before calling the woe model.

## Full working example
```
from woe_conversion.woe import *
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

titanic = sns.load_dataset('titanic')
train, test = train_test_split(titanic, test_size=0.30,random_state=111,stratify=titanic['survived'])

woemodel = WoeConversion(binarytarget='survived', features=['age','sex','class','fare'])
woemodel.fit(train)
transformedtrain = woemodel.transform(train)
transformedtest = woemodel.transform(test)

clf = LogisticRegression()
clf.fit(transformedtrain[['age','sex','class','fare']], transformedtrain['survived'])

transformedtrain['proba'] = clf.predict_proba(transformedtrain[['age','sex','class','fare']])[:,1]
train['proba'] = clf.predict_proba(transformedtrain[['age','sex','class','fare']])[:,1]
transformedtest['proba'] = clf.predict_proba(transformedtest[['age','sex','class','fare']])[:,1]
test['proba'] = clf.predict_proba(transformedtest[['age','sex','class','fare']])[:,1]

```




## Author
woe was created by Bertrand Brelier. If you have any questions or issues, please feel free to contact the author.
