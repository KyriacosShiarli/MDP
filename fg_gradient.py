from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

def adaboost_class(x,y,estimators,depth):
	bst = AdaBoostClassifier(DecisionTreeClassifier(max_depth = depth),algorithm = "SAMME.R",n_estimators = estimators)
	bst.fit(x,y)
	return bst

def adaboost_reg(x,y,estimators,depth):
	reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth = depth),n_estimators = estimators)
	reg.fit(x,y)
	return reg