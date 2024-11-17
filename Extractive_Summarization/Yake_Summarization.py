import pandas as pd
import yake


#輸入新聞領域(Technology, Business, Education, Politics, Celebrity_Full, Entertainment, Complete_FakeNews, Sports)
domain = input("Please input the domain:") 

file_path = f"./Dataset/TALLIP-FakeNews-Dataset/English/Train/train_English_Data_{domain}.txt"
news_data = pd.read_csv(file_path, sep="\t")

# 設定 Yake 的參數
language = "en"        # 語言
max_ngram_size = 3     # 最大的 N-gram 長度
deduplication_threshold = 0.9  # 去重閾值，範圍為 0~1，值越小越容易過濾相似詞。
num_of_keywords = 5    # 提取的關鍵字數量，若沒設定，預設會返回最多 20 個關鍵字。

# 初始化 Yake 設定
kw_extractor = yake.KeywordExtractor(
    lan=language,
    n=max_ngram_size,
    dedupLim=deduplication_threshold,
    top=num_of_keywords
)

# 定義關鍵字提取函數
def extract_keywords(text):
    keywords = kw_extractor.extract_keywords(text)
    return [kw for kw, score in keywords]

news_data['Keywords'] = news_data['News'].apply(extract_keywords)


# print(news_data[['News', 'Keywords']])
print(news_data)
# news_data.to_csv(f"./keyword_result/{domain}_keyword.csv", index=False)
