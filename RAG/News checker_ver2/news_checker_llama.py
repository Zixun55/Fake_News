from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.callbacks import StreamingStdOutCallbackHandler

# model_name = 'meta-llama/Llama-2-7b-chat-hf'
model_name = './fine_tuned_model_ver2_merged'

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

def classify_fake_news(news, explanation):
    prompt = f"""This is a news article: {news}
                 Explanation: {explanation}
                 Based on the explanation above, determine whether this news is true news or false news.\n\n"""

    response = llm.invoke(prompt)
    return response

news_article = "Gubernatorial candidate Kelly Ayotte “has voted for national abortion bans, she has voted to defund Planned Parenthood, and she shepherded Neil Gorsuch through the Supreme Court process, and then celebrated when Roe v. Wade was overturned."
news_explanation = ("It's misleading to say that as a U.S. senator, Kelly Ayotte voted for a national abortion ban. Twice, Ayotte supported a bill to end abortions after 20 weeks, with exceptions after that for the mother's life and physical health, rape, and incest."
                    "Ayotte voted several times in the Senate to defund Planned Parenthood, and she later served as the 'sherpa' for Neil Gorsuch's Supreme Court justice confirmation in the Senate. "
                    "Ayotte applauded the Supreme Court's decision to hand responsibility for abortion law to the states.")

result = classify_fake_news(news_article, news_explanation)
print(result)