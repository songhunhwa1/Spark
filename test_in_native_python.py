from sklearn import preprocessing

from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

# df
df = pd.read_csv("/Users/woowahan/Documents/Python/Pyspark_MLlib/data/Default.csv", index_col=0)

# preprocessing
le = preprocessing.LabelEncoder()
df['default_idx'] = le.fit_transform(df.default)
df['student_idx'] = le.fit_transform(df.student)

df.drop(["default", "student"], axis=1, inplace=True)

# split
X = df.drop("default_idx", axis=1)
y = df.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)

# MODEL
def pipeline_LR(X_train, y_train):
    
    # FIXME: 최소값을 0, 최대값을 1로 변환하여 scale
    scale = preprocessing.MinMaxScaler(feature_range = [0,1])
    
    # FIXME: 높은 설명력 기준으로 변수의 수 설정
    selectK = SelectKBest(score_func=f_classif)
    
    # FIXME: 모델 fit
    logistic_model = LogisticRegression()
    
    # 파이프라인 구성
    pipeline_object = Pipeline([('feature_selection', selectK), ('model', logistic_model)])
    
    # FIXME: Parmeters 튜닝
    params = [{'feature_selection__k': ['all'],
               'model__C': [0.01, 0.1, 1, 5, 10],
               'model__penalty': ['l2'],
               'model__solver': ['lbfgs', 'liblinear', 'sag']
              }] 
    
    # FIXME: AUC 기준 최적 Params 도출 / 10개의 k-fold 설정
    grid_search = GridSearchCV(pipeline_object, param_grid = params, scoring = 'roc_auc', cv=10)
    
    # Model fit
    grid_search.fit(X_train.values, y_train.values)
    print("Accuracy :", grid_search.best_score_)
    return grid_search

#
model_LR = pipeline_LR(X_train, y_train)

model_LR.best_estimator_

preds = model_LR.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_true = y_test, y_pred=preds))
print(pd.crosstab(preds, y_test, rownames=['Predicted'], colnames=['True']))
print(metrics.classification_report(y_test, preds))

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, preds)
roc_auc = auc(false_positive_rate, true_positive_rate)

print(roc_auc)








