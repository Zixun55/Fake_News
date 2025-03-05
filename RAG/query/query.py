from transformers import pipeline
import nltk
from nltk.tokenize import sent_tokenize

# nltk.download('punkt')
# nltk.download('punkt_tab')

# 載入LLM Model
# llm = pipeline("summarization", model="t5-small")
# llm = pipeline("summarization", model="t5-base")
# llm = pipeline("summarization", model="facebook/bart-base")  # 效果目測不太好，常常只截半句話
llm = pipeline("summarization", model="google/pegasus-xsum")



# 新聞文章
# news_article = "Gubernatorial candidate Kelly Ayotte “has voted for national abortion bans, she has voted to defund Planned Parenthood, and she shepherded Neil Gorsuch through the Supreme Court process, and then celebrated when Roe v. Wade was overturned.”"
news_article = "England 's all - time leading wicket - taker Jimmy Anderson remains confident England 's all - time leading wicket - taker Jimmy Anderson body can withstand the rigours of seven Tests in two months this summer but   as England 's all - time leading wicket - taker Jimmy Anderson turns 35 in July   accepts England 's all - time leading wicket - taker Jimmy Anderson has to take things \" step by step \" . As a result   England 's all - time leading wicket - taker Jimmy Anderson is unable to look as far ahead as England 's tour of Australia later in the year   which would be England 's all - time leading wicket - taker Jimmy Anderson fourth Ashes tour . Since missing England 's final Test in India in December – the fifth Test England 's all - time leading wicket - taker Jimmy Anderson has missed through injury in England 's past 11 – with the problematic shoulder that affected England 's all - time leading wicket - taker Jimmy Anderson year   England 's all - time leading wicket - taker Jimmy Anderson spent January resting and has since undergone a full pre‑season with Lancashire England 's all - time leading wicket - taker Jimmy Anderson ."

# 分句
sentences = sent_tokenize(news_article)

# 提取關鍵句或摘要
query_sentences = []
for sentence in sentences:
    summary = llm(sentence, max_length=30, min_length=5, do_sample=True, top_p=0.85)
    query_sentences.append(summary[0]['summary_text'])

# 合併為 Query
query = " ".join(query_sentences)
print("Generated Query:", query)
