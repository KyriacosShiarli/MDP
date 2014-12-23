from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor


def adaboost_class(x,y,estimators,depth):
	bst = AdaBoostClassifier(DecisionTreeClassifier(max_depth = depth),algorithm = "SAMME.R",n_estimators = estimators)
	bst.fit(x,y)
	return bst

def adaboost_reg(x,y,estimators,depth):
	alpha = 0.9
	reg = GradientBoostingRegressor(loss='huber', alpha=alpha,
                                n_estimators=300, max_depth=15,subsample = 0.1,
                                learning_rate=0.1, min_samples_leaf=2,
                                min_samples_split=5,verbose=0)
	reg.fit(x,y)
	return reg


