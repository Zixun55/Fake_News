import pandas as pd
from summa import keywords

###資料讀取###
#使用TALLIP資料集
#輸入新聞領域(Technology, Business, Education, Politics, Celebrity_Full, Entertainment, Complete_FakeNews, Sports)
domain = input("Please input the domain:") 
file_path = f"./Dataset/TALLIP-FakeNews-Dataset/English/Train/train_English_Data_{domain}.txt"
news_data = pd.read_csv(file_path, sep="\t")

#使用ISOT Fake News detection dataset
# file_path = f"./Dataset/archive/True.csv"
# news_data = pd.read_csv(file_path)
# news_data = news_data.head(300)

###提取關鍵字###
# 定義一個函數來提取關鍵字
def extract_keywords(text):
    # return keywords.keywords(text, ratio=0.2).split("\n")
    return keywords.keywords(text, words=10).split("\n")

###使用哪個資料集進行關鍵字提取###
# 適用於 'News' 欄位，提取每篇新聞的關鍵字
news_data['Keywords'] = news_data['News'].apply(extract_keywords)
# news_data['Keywords'] = news_data['text'].apply(extract_keywords)

###輸出結果###
print(news_data)
news_data.to_csv(f"./keyword_result/{domain}_keyword_text_rank.csv", index=False)
# news_data.to_csv(f"./keyword_result/ISOT_keyword_text_rank.csv", index=False)