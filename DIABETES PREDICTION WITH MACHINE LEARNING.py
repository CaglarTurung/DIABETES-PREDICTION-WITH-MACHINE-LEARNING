import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split, cross_validate
from helpers.data_prep import *
from helpers.eda import *



pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


df = pd.read_csv("datasets/diabetes.csv")
df.head()
df.describe().T
check_df(df)

######################################################################################################################################################

# Modelling before data preprocess
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)
log_model = LogisticRegression(max_iter=100000).fit(X, y)
cv_results = cross_validate(log_model,
                            X, y,
                            cv=10,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
# Accuracy: 0.7734791524265209

cv_results['test_precision'].mean()
# Precision: 0.7283896804949437

cv_results['test_recall'].mean()
# Recall:  0.5709401709401709

cv_results['test_f1'].mean()  # Recall deyerinin kotu olmasi f1 skoruna yansidi
# F1-score: 0.6377596026885664

cv_results['test_roc_auc'].mean()
# AUC: 0.8294188034188034

y_pred = log_model.predict(X)

# Confusion Matrix
def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(y, y_pred)
######################################################################################################################################################

#FEATURE ENGINEERING
# Feature engineering was prepared in a different file and saved as a pickle file.
df = pd.read_pickle("prepared_diabetes_df.pkl")
df.head()

#MODEL
y = df["OUTCOME"]
X = df.drop(["OUTCOME"], axis=1)

log_model = LogisticRegression().fit(X, y)

log_model.intercept_ #---> B bias
log_model.coef_     #----> W weight

y_pred = log_model.predict(X)

y_pred[0:10]
y[0:10]

# ROC AUC
y_prob = log_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob) # 0.87

# MODEL VALIDATION METHODS

######################################################
# Model Validation: 5-Fold Cross Validation
######################################################

y = df["OUTCOME"]
X = df.drop(["OUTCOME"], axis=1)

log_model = LogisticRegression().fit(X, y)


cv_results = cross_validate(log_model,
                            X, y,
                            cv=5,
                            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])



cv_results['test_accuracy'].mean()
# Accuracy: 0.77

cv_results['test_precision'].mean()
# Precision: 0.77

cv_results['test_recall'].mean()
# Recall: 0.78

cv_results['test_f1'].mean()
# F1-score: 0.77

cv_results['test_roc_auc'].mean()
# AUC: 0.86



######################################################
# Prediction for A New Observation
#####################################################

X.columns

random_user = X.sample(1, random_state=45)

log_model.predict(random_user) # 0

