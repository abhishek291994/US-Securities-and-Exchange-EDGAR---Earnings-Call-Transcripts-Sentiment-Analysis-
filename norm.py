import pandas as pd
from sklearn import preprocessing
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# Read file and normalize all api scores within same range
df = pd.read_csv("norm.csv")
names = df.columns
print(names)

# Create a scalar object
scaler = preprocessing.MinMaxScaler()


# Fit scores data on the scaler object
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns=names)

#print(scaled_df)

scaled_df['Normalized_Score'] = scaled_df.mean(axis=1)

print(scaled_df)

# Create label from normalized score
def new_label(row):
    if row['Normalized_Score'] >= 0.6:
        label = 1
    elif row['Normalized_Score'] >= 0.4 and  row['Normalized_Score'] <0.6:
        label = 0
    else:
        label = 2
    return label	



label = scaled_df.apply(new_label,axis=1)
scaled_df['score_label'] = label

print(scaled_df['score_label'])

export_csv = scaled_df.to_csv (r'/home/shravs/deep_learning/sentiment/normalized_op.csv', index = None, header=True) 

# Get the manually labelled scores
true = pd.read_csv("API_Scores.csv")

# Confusion matrix 
cm = confusion_matrix(true['Manual_label'],scaled_df['score_label'])

#print(cm)

LABELS = ['negative', 'positive','neutral']
sns.heatmap(cm, annot=True, xticklabels=LABELS, yticklabels=LABELS, fmt='g')
xl = plt.xlabel("Predicted")
yl = plt.ylabel("Actual")


