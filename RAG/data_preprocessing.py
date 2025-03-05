import pandas as pd
from sklearn.model_selection import train_test_split

# 讀取資料
data = pd.read_csv('./knowledge_base/clean_data/politifact_clean_original.csv')

data = data[['Claim', 'Explanation', 'Classification']]

label_map = {
    'true': 1,
    'mostly-true': 1,
    'half-true': 0,
    'barely-true': 0,
    'false': 0,
    'pants-fire': 0
}

# data['Classification'] = data['Classification'].map(label_map)
### 我在想可能map出的型別是string不是int ###
data['Classification'] = data['Classification'].map(label_map).astype(int)

train_data, test_data = train_test_split(data, train_size=0.8, random_state=42)

train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)
