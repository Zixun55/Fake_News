from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.callbacks import StreamingStdOutCallbackHandler
import pandas as pd
import re
import torch

# model_name = './fine_tuned_llama2'  # 訓練過的模型路徑
model_name = 'meta-llama/Llama-2-7b-chat-hf'  # 可以選擇不同的版本

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto', device_map='auto')

text_pipeline = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.2
)

callbacks = [StreamingStdOutCallbackHandler()]

llm = HuggingFacePipeline(pipeline=text_pipeline, callbacks=callbacks, verbose=False)

df = pd.read_csv('./dataset/True_ordered.csv')
# news_articles = df[['id', 'text']].iloc[2816:2817] # id = 2817 (test max_token exception)
# news_articles = df[['id', 'text']].iloc[8338:8339] # id = 8339 (test out_of_memory exception)

news_articles = df[['id', 'text']].head(10) # id = 1~6000 (6000)
# news_articles = df[['id', 'text']].iloc[0:12000] # id = 6001~12000 (12000)
# news_articles = df[['id', 'text']].iloc[12000:18000] # id = 12001~18000 (18000)
# news_articles = df[['id', 'text']].iloc[18001:21417] # id = 18001~21417 (21417)
# news_articles = df[['id', 'text']].iloc[18001:23481] # id = 18001~23481 (23481)

generated_queries = []

def generate_query(news): 
    prompt = f'''You are extracting a search query from a news article to retrieve the most relevant information from a knowledge base.

    Input:
    News Article: {news}

    Task:
    Based on the provided news article, generate one clear, complete question that accurately represents the main topic. 
    The question should be specific, relevant, informative.
    Additionally, extract only 5 essential keywords that summarize the main topics of the article.

    Output:
    1. Query: (A concise, meaningful question)
    2. Keywords: (keyword 1, keyword 2, keyword 3, keyword 4, keyword 5)

    Ensure that your response strictly follows the provided format.\n\n'''
    
    try:
        response = llm.invoke(prompt)
        return response

    except ValueError as e:
        if 'max_length' in str(e):
            return 'format.\n\nQuery: max_token?\nKeywords: max_token'
        else:
            raise e
        
    except RuntimeError as e:
        if 'CUDA out of memory' in str(e):
            torch.cuda.empty_cache()
            return 'format.\n\nQuery: out_of_memory?\nKeywords: out_of_memory'
        else:
            raise e

def extract_generated_query(response_text):
    output = response_text.split('format.\n\n')
    result = output[1].strip()
    
    return result

def split_generated_query(response_text):
    # get Query: ~ ?
    query_match = re.search(r'Query:\s*(.*\?)', response_text)
    
    if query_match:
        query = query_match.group(1).strip()
        
        # count '?' == 1
        if query.count('?') != 1:
            query = ''
            print('問號數量錯誤')
        else:
            query = query.strip('\"')
            query = f'"{query}"'
    else:
        query = ''
        print('找不到Query')
    
    # get Keywords:
    keywords_match = re.search(r'Keywords:\s*(.*)', response_text, re.DOTALL)
    
    if keywords_match:
        keywords = keywords_match.group(1).strip()
        
        # allow A-Z, a-z, 0-9, space,  [' " , - % $ .], and not allow \n
        if not re.match(r'^[A-Za-z0-9\s,\'\"\-\%\.\$\+\/\&]+$', keywords) or '\n' in keywords:
            keywords = ''
            print('關鍵字格式錯誤')
            
        # count ',' == 4 (5 keywords)  
        elif keywords.count(',') != 4:
            keywords = ''
            print('關鍵字數量錯誤')
            
    else:
        keywords = ''
        print('找不到Keywords')

    return query, keywords

for index, row in news_articles.iterrows():
    news_id = row['id']
    news_article = row['text']
    
    generated_query = generate_query(news_article)
    generated_query = extract_generated_query(generated_query)
        
    query, keywords = split_generated_query(generated_query)

    print(f'新聞 ID: {news_id}\n')
    print(generated_query)
    
    generated_queries.append({'id': news_id, 'Query': query, 'Keywords': keywords})

df_result = pd.DataFrame(generated_queries)
df_result.to_csv('generated_queries_faketest.csv', index=False, encoding='utf-8-sig')