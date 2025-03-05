from transformers import LlamaForCausalLM, LlamaTokenizer
import torch

# 設置模型名稱（根據 Hugging Face 的 LLaMA 頁面）
model_name = "meta-llama/Llama-2-7b-chat-hf"  # 可以選擇不同的版本

# 加載 tokenizer 和模型
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

# 設置模型為評估模式
model.eval()

device = torch.device("cpu")
model.to(device)

news_article = "During her tenure as a U.S. Senator, Kelly Ayotte voted to defund Planned Parenthood and supported a 20-week national abortion ban. She also played a key role in the confirmation of Supreme Court Justice Neil Gorsuch, whose appointment contributed to the overturning of Roe v. Wade. In her 2024 gubernatorial campaign, Ayotte pledged to uphold New Hampshire's existing abortion laws, allowing procedures up to 24 weeks, and promised to veto any further restrictions."
# news_article = "House Speaker “Kevin McCarthy has quietly implemented a pay raise for members (of Congress) that could be $30,000+ per person. It circumvents the Constitution by instead reimbursing their rent, utilities, & meals.”"

# 設定生成查詢的提示
# query_prompt = (
#     f"""Extract concise keywords from the following news article and concatenate them into a comma-separated string.
#     News Article: {news_article}
#     Keywords: """
# )
query_prompt = (
    f"以下是一篇新聞文章，請從中提取數個簡短的關鍵詞，並將這些關鍵字用空格分隔串起來形成一個字串，不要有任何符號、條列式或其他多餘的格式，也不要有*號。這些關鍵詞應該幫助查找相關資料以判斷文章的真實性。\n"
    # f"以下是一篇新聞文章，請提取一個簡短且具體的查詢問句，這個查詢問句是之後我要在外部網站查找相關資料以判斷文章的真實性。你只需要給我生成的查詢問句就好了，不要添加任何額外的陳述句子或說明。\n"
    f"{news_article}\n"
    f"查詢："
)

# 編碼輸入文本
inputs = tokenizer(query_prompt, return_tensors="pt")
inputs = {key: value.to(device) for key, value in inputs.items()}

# 生成查詢
with torch.no_grad():
    outputs = model.generate(
        inputs["input_ids"],
        # max_new_tokens=100,  # 限制生成文字數
        num_beams=5,
        early_stopping=True,
        repetition_penalty=1.2
        # temperature=0.7
    )

# 解碼生成的文本
generated_query = tokenizer.decode(outputs[0])
if "查詢：" in generated_query:
    generated_query = generated_query.split("查詢：", 1)[-1].strip()

generated_query = generated_query.split("\n")[0].strip()

# 顯示結果
print(f"{generated_query}")