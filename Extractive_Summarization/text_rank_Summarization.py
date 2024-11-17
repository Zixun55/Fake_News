import pandas as pd
from summa import keywords

#輸入新聞領域(Technology, Business, Education, Politics, Celebrity_Full, Entertainment, Complete_FakeNews, Sports)
domain = input("Please input the domain:") 

file_path = f"./Dataset/TALLIP-FakeNews-Dataset/English/Train/train_English_Data_{domain}.txt"
news_data = pd.read_csv(file_path, sep="\t")

# 定義一個函數來提取關鍵字
def extract_keywords(text):
    return keywords.keywords(text).split("\n")

# 適用於 'News' 欄位，提取每篇新聞的關鍵字
news_data['Keywords'] = news_data['News'].apply(extract_keywords)

# 顯示結果
print(news_data)