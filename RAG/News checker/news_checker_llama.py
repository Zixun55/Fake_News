from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.callbacks import StreamingStdOutCallbackHandler

model_name = 'meta-llama/Llama-2-7b-chat-hf'
# model_name = 'meta-llama/Llama-2-7b-hf'
# model_name = 'meta-llama/Meta-Llama-3-8B'

# model_name = './fine_tuned_model'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto', device_map='auto')

text_pipeline = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    # max_new_tokens=100,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.2
)

callbacks = [StreamingStdOutCallbackHandler()]

llm = HuggingFacePipeline(pipeline=text_pipeline, callbacks=callbacks, verbose=True)

def classify_fake_news(news, claim, explanation, ground_truth):
    # prompt = f"""This is a news article: {news}
    # Explanation: {explanation}
    # Ground Truth: {ground_truth}a
    # Based on the explanation above, determine whether the news article is true or false and provide a confidence score.\n\n"""
    
    prompt = f"""You are analyzing the authenticity of a news article.  

        News Article: {news}  
        
        Retrieved Relevant Information:
        Claim: {claim}  
        Explanation: {explanation}  
        Ground Truth: {ground_truth}  

        Task:  
        Based on the retrieved claim, its explanation, and the ground truth, determine whether the given news article is true or false. Consider the relevance and credibility of the retrieved information when making your decision.  

        Provide:
        1. **Classification**: (True/False)  
        2. **Confidence Score**: (0-100)  
        3. **Justification**: (Brief explanation referencing the retrieved explanation and ground truth)
        
        Ensure that your response strictly follows the provided format.\n\n"""
    
    response = llm.invoke(prompt)
    return response

news_article = "During her tenure as a U.S. Senator, Kelly Ayotte voted to defund Planned Parenthood and supported a 20-week national abortion ban. She also played a key role in the confirmation of Supreme Court Justice Neil Gorsuch, whose appointment contributed to the overturning of Roe v. Wade. In her 2024 gubernatorial campaign, Ayotte pledged to uphold New Hampshire's existing abortion laws, allowing procedures up to 24 weeks, and promised to veto any further restrictions."
claim = "Gubernatorial candidate Kelly Ayotte “has voted for national abortion bans, she has voted to defund Planned Parenthood, and she shepherded Neil Gorsuch through the Supreme Court process, and then celebrated when Roe v. Wade was overturned."
news_explanation = ("It's misleading to say that as a U.S. senator, Kelly Ayotte voted for a national abortion ban. Twice, Ayotte supported a bill to end abortions after 20 weeks, with exceptions after that for the mother's life and physical health, rape, and incest."
                    "Ayotte voted several times in the Senate to defund Planned Parenthood, and she later served as the 'sherpa' for Neil Gorsuch's Supreme Court justice confirmation in the Senate. "
                    "Ayotte applauded the Supreme Court's decision to hand responsibility for abortion law to the states.")
classification = 'false'

result = classify_fake_news(news_article, claim, news_explanation, classification)
print(result)