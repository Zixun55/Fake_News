from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.callbacks import StreamingStdOutCallbackHandler

model_name = 'meta-llama/Llama-2-7b-chat-hf'

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto', device_map='auto')

text_pipeline = pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.2
)

callbacks = [StreamingStdOutCallbackHandler()]

llm = HuggingFacePipeline(pipeline=text_pipeline, callbacks=callbacks, verbose=True)

response = llm.invoke('Hi')
print(response)