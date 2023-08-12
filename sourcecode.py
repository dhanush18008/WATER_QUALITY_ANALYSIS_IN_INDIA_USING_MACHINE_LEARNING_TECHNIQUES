import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import missingno as msno
from scipy import stats
sett = pd.read_csv('/content/water_potability.csv')
sett
sett.info()
sett.describe().style.background_gradient(cmap = "Blues")
sett.duplicated().sum()
msno.bar(sett, figsize = (16,5),color = "#483D8B")
plt.show()
for column_name in sett.columns:
    unique_values = len(sett[column_name].unique())
    print("Feature '{column_name}' has '{unique_values}' unique values".format(column_name = column_name,
                                                                                         unique_values=unique_values))
sett.columns
numeric_features=['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity',
       'Organic_carbon', 'Trihalomethanes', 'Turbidity']
import warnings
warnings.filterwarnings('ignore')
fig,ax = plt.subplots(len(numeric_features),4,figsize=(30,20))
for index,i in enumerate(numeric_features):
    sns.distplot(sett[i],ax=ax[index,0],color='green')
    sns.boxplot(sett[i],ax=ax[index,1],color='yellow')
    sns.violinplot(sett[i],ax=ax[index,2],color='purple')
    stats.probplot(sett[i],plot=ax[index,3])
    
fig.tight_layout()
fig.subplots_adjust(top=0.95)
plt.suptitle("Visualizing continuous columns (sett dataset)",fontsize=30)
fig, ax1 = plt.subplots(figsize=(20,10))
graph = sns.countplot(ax=ax1,x = 'Potability' , data = sett,palette='pastel')
graph.set_xticklabels(graph.get_xticklabels(),rotation=90)
for p in graph.patches:
    height = p.get_height()
    graph.text(p.get_x()+p.get_width()/2., height + 0.1,height ,ha="center")
sett.isnull().sum()

from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=10, weights="uniform")
l=imputer.fit_transform(sett)
sett1=pd.DataFrame(l,columns=sett.columns)
sett1
sett1.isnull().sum()
# Shapiro-Wilk Test
from scipy.stats import shapiro
# normality test
for feature in numeric_features:
	stat, p = shapiro(sett1[feature])
	print('Statistics=%.3f, p=%.3f' % (stat, p))
	# interpret
	alpha = 0.05
	if p > alpha:
		print(f'Sample looks Gaussian with {feature} (fail to reject H0)')
	else:
		print(f'Sample does not look Gaussian with {feature} (reject H0)')

from scipy.stats import mannwhitneyu
for feature in numeric_features:
    stat, p = mannwhitneyu(sett1['Potability'], sett1[feature])
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    alpha = 0.05
    if p > alpha:
		    print('There are no significant differences (fail to reject H0)')
    else:
		    print('The sample distributions are not equal (reject H0)')
correlation = sett1.corr()
print(correlation['Potability'].sort_values(ascending = False),'\n')
k= 18
cols = correlation.nlargest(k,'Potability')['Potability'].index
print(cols)
cm = np.corrcoef(sett1[cols].values.T)
mask = np.triu(np.ones_like(sett1.corr()))
f , ax = plt.subplots(figsize = (14,12))
sns.heatmap(cm,mask=mask, vmax=.8, linewidths=0.01,square=True,annot=True,cmap='viridis',
            linecolor="white",xticklabels = cols.values ,annot_kws = {'size':12},yticklabels = cols.values)
from sklearn.neighbors import LocalOutlierFactor
clf = LocalOutlierFactor(n_neighbors=5, contamination='auto')
y_pred = clf.fit_predict(sett1) 
sett1['Out']=y_pred
Out=sett[sett1['Out']!=1]
Out.shape
sett2=sett1[sett1['Out']==1]
sett2=sett2.drop('Out',axis=1)
from imblearn.over_sampling import SMOTE 
oversample = SMOTE()
features, labels=  oversample.fit_resample(sett2.drop(["Potability"],axis=1),sett2["Potability"])
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
names = features.columns
d = scaler.fit_transform(features)

scaled_df = pd.DataFrame(d, columns=names)
scaled_df.head()
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb

from sklearn.metrics import precision_score,recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix,accuracy_score, classification_report
X_sett, X_test, y_sett, y_test=train_test_split(scaled_df,labels,test_size=0.33,random_state=42)
models = [RandomForestClassifier(), KNeighborsClassifier(), SVC(), LogisticRegression(),xgb.XGBClassifier()]
scores = dict()

for m in models:
    m.fit(X_sett, y_sett)
    y_pred = m.predict(X_test)

    print(f'model: {str(m)}')
    print(classification_report(y_test,y_pred, zero_division=1))
    print('-'*30, '\n')
!pip install optuna
import optuna 
from optuna import Trial, visualization
from optuna.samplers import TPESampler
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
def objective(trial: Trial,X,y) -> float:
    
    param = {
            'n_estimators' : trial.suggest_int("n_estimators",100,1500),
            'max_features' : trial.suggest_categorical("max_features",["auto","sqrt"]),
            'max_depth' : trial.suggest_int("max_depth",5,80,log = True),
            'min_samples_split' : trial.suggest_int("min_samples_split",2,15),
            'min_samples_leaf' : trial.suggest_int("min_samples_leaf",1,9),
            'bootstrap' : trial.suggest_categorical("bootstrap",[True,False])
            }
    
    model = RandomForestClassifier(**param)
    
    return cross_val_score(model, X, y, cv=5).mean()
study = optuna.create_study(direction='maximize',sampler=TPESampler())
study.optimize(lambda trial : objective(trial,X_sett,y_sett),n_trials= 20)
from optuna import visualization
print('Best trial: CV_score= {},\nparams {}'.format(study.best_trial.value,study.best_trial.params))
hist = study.trials_dataframe()
hist.head()
optuna.visualization.plot_slice(study)
optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_parallel_coordinate(study)
optuna.visualization.plot_param_importances(study)
Best_trial = study.best_trial.params
best_clf2=RandomForestClassifier(**Best_trial)
best_clf2.fit(X_sett, y_sett)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(best_clf2.predict(X_test),y_test)
disp = ConfusionMatrixDisplay(cm, display_labels=["0","1"])
disp.plot()
plt.title("Confusion Matrix")
plt.show()
feature_importances=best_clf2.feature_importances_
feature_importances_df=pd.DataFrame({'features':list(X_sett), 'feature_importances':feature_importances})
feature_importances_df=pd.DataFrame({'features':list(X_sett), 'feature_importances':feature_importances})
feature_importances_df.sort_values('feature_importances',ascending=False)