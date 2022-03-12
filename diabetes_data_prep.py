import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from helpers.eda import *
from helpers.data_prep import *

pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)

def load():
    data = pd.read_csv(r"E:\CAGLAR\datasets\diabetes.csv")
    return data

df = load()

# DESCRIPTIVE STATISTICS

df.shape
df.info()
df.columns
df.describe().T # It is looks like there are outlier values in the Insulin column.

# Are there any missing values in the dataset ?
df.isnull().values.any()  # There are no missing values in the data set.

check_df(df)

df.describe().T

# CONFUSION MATRIX
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)
log_model = LogisticRegression(max_iter=100000).fit(X, y)
y_pred = log_model.predict(X)
def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size=10)
    plt.show()

plot_confusion_matrix(y, y_pred)
# Looking at the confusion matrix, the dataset looks like an unbalanced data set.
cat_summary(df, "Outcome", ratio=True, plot=True) # When we look at it with the cat_summary function, it is clear that there is an imbalance in the data set.

#We try to fix it with SMOTE that is one of the oversampling methods.
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)
oversample = SMOTE()
X, y = oversample.fit_resample(X, y)
df = pd.concat([X, y], axis=1)
df.head()

# LETS CHECK CAT_SUMMARY AGAIN
cat_summary(df, "Outcome", ratio=True, plot=True) # UNBALANCED DATA HAS FIXED.


# FEATURE ENGINEERING
df.columns = [col.upper() for col in df.columns]

def diabets_data_prep(dataframe):
    dataframe["NEW_AGE_PREG"] = dataframe["AGE"] * dataframe["PREGNANCIES"]
    dataframe["NEW_AGE_SKIN"] = dataframe["AGE"] * dataframe["SKINTHICKNESS"]
    dataframe["NEW_PREG_PEDİGRE"] = dataframe["PREGNANCIES"] * dataframe["DIABETESPEDIGREEFUNCTION"]

    dataframe.loc[(dataframe["BMI"] < 18.5 ), "NEW_BMI_CAT"] = "underweight"
    dataframe.loc[(dataframe["BMI"] >= 18.5) & (dataframe["BMI"] < 25), "NEW_BMI_CAT"] = 'healthyweight'
    dataframe.loc[(dataframe["BMI"] >= 25) & (dataframe["BMI"] < 30), "NEW_BMI_CAT"] = 'aboveweight'
    dataframe.loc[(dataframe["BMI"] >= 30), "NEW_BMI_CAT"] = 'obesite'

    dataframe.loc[(dataframe["AGE"] >= 18) & (dataframe["AGE"] < 56), "NEW_AGE_CAT"] = "mature"
    dataframe.loc[(dataframe["AGE"] >= 56), "NEW_AGE_CAT"] = "senior"

    dataframe.loc[(dataframe["BLOODPRESSURE"] >= 110 ), "NEW_BLOOD_CAT"] = "emergency"
    dataframe.loc[(dataframe["BLOODPRESSURE"] < 110) & (dataframe["BLOODPRESSURE"] >= 100 ), "NEW_BLOOD_CAT"] = "risky"
    dataframe.loc[(dataframe["BLOODPRESSURE"] < 100) & (dataframe["BLOODPRESSURE"] >= 90), "NEW_BLOOD_CAT"] = "important"
    dataframe.loc[(dataframe["BLOODPRESSURE"] < 90) & (dataframe["BLOODPRESSURE"] >80), "NEW_BLOOD_CAT"] = "warning"
    dataframe.loc[(dataframe["BLOODPRESSURE"] <= 80), "NEW_BLOOD_CAT"] = "normal"

    dataframe.loc[(dataframe["GLUCOSE"] >= 199), "NEW_GLUCOSE_CAT"] = "diabetes"
    dataframe.loc[(dataframe["GLUCOSE"] < 199) & (dataframe["GLUCOSE"] >= 141), "NEW_GLUCOSE_CAT"] = "pre-diabetes"
    dataframe.loc[(dataframe["GLUCOSE"] <= 140), "NEW_GLUCOSE_CAT"] = "normal"

    dataframe.loc[(dataframe["INSULIN"] >= 166), "NEW_INSULIN_CAT"] = "high"
    dataframe.loc[(dataframe["INSULIN"] < 166) & (dataframe["INSULIN"] >= 16), "NEW_INSULIN_CAT"] = "normal"
    dataframe.loc[(dataframe["INSULIN"] < 16), "NEW_INSULIN_CAT"] = "low"

    return dataframe

df_prep = diabets_data_prep(df)
df_prep.head()

#REPLACE WITH TRESHOLDS
    cat_cols, num_cols, cat_but_car = grab_col_names(df_prep)
    cat_cols.remove("OUTCOME")

    num_cols = [col for col in df_prep.columns if len(df_prep[col].unique()) > 20
                and df_prep[col].dtypes != 'O']

    for col in num_cols:
        replace_with_thresholds(df_prep, col)

check_df(df_prep)

#BINARY ENCODING
    binary_cols = [col for col in df_prep.columns if
                   len(df_prep[col].unique()) == 2 and df_prep[col].dtypes == 'O']

    for col in binary_cols:
        df_prep = label_encoder(df_prep, col)

#ONE-HOT ENCODER
    ohe_cols = [col for col in df_prep.columns if 10 >= len(df_prep[col].unique()) > 2]
    df_prep = one_hot_encoder(df_prep, ohe_cols)

df_prep.head()
df_prep.shape

# ROBUST SCALER

rs = RobustScaler()
cat_cols, num_cols, cat_but_car = grab_col_names(df_prep)

df_prep[num_cols] = rs.fit_transform(df_prep[num_cols])
df_prep.head()
df_prep.shape

df_prep.to_pickle("prepared_diabetes_df.pkl") # the .py file was converted to pkl file.


