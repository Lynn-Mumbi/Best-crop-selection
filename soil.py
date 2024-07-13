# importing packages
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.metrics import f1_score

# Load the dataset
crops = pd.read_csv("soil_measures.csv")

#print first lines of dataset
print(crops.head())
#number of crops
print("There are", crops["crop"].nunique(), "crops")
#checking for missing values
print(crops.isna().sum())   # no missing data
#data types
print(crops.info())

#splitting the data into training and test sets.
X_train,X_test,y_train,y_test=train_test_split(crops[["N","P","K","ph"]],crops['crop'],test_size=0.2,random_state=42)

# predicting the crop
f1_scores = []
# Loop over each feature index
for feature in ["N", "P", "K", "ph"]:
    # Initialize and fit logistic regression model
    log_reg = LogisticRegression(max_iter=2000, multi_class="multinomial")
    log_reg.fit(X_train[[feature]], y_train)
    # predicting test set
    y_pred = log_reg.predict(X_test[[feature]])
    # getting f1 score
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"{feature} f1 score is the f1 score for feature {feature} : {f1}")

# performing correlation analysis by calculating the correlation matrix
correlation_matrix = crops[["N","P","K","ph"]].corr()

# Print the correlation matrix
print("Correlation Matrix:")
print(correlation_matrix)
sns.heatmap(crops[["N","P","K","ph"]].corr(),annot=True)
plt.show()

# Extracting final features
X_final = ['N', 'K', 'ph']

# Splitting the data into training and test sets
X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(crops[X_final], crops["crop"], test_size=0.2, random_state=42)

# Initializing,train and fit logistic regression model
log_reg = LogisticRegression(max_iter=2000, multi_class="multinomial")
log_reg.fit(X_train_final,y_train_final)

# Predicting on the test set
y_pred_final= log_reg.predict(X_test_final)

# model_performance is as shown using f1_score:
model_performance = f1_score(y_test_final,y_pred_final,average="weighted")
print(model_performance)