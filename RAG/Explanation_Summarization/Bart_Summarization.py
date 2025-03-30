import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 讀取 CSV
df = pd.read_csv("./knowledge_base/original_data/snopes_mixture2.csv")

# 選擇 T5-Large 模型
MODEL_NAME = "google/t5-v1_1-large"

# 加載 Tokenizer 和 Model（使用 FP16 降低 VRAM 需求）
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to("cuda")

# 定義摘要函數
def summarize_text(text, max_len=150, min_len=50):
    if pd.isna(text) or len(text.strip()) == 0:
        return text
    try:
        # 前處理：T5 需要加上 "summarize: " 前綴
        input_text = "summarize: " + text

        # 檢查 Token 數量，避免超過 4096 限制
        input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=4096).to("cuda")

        # 產生摘要
        summary_ids = model.generate(
            input_ids,
            max_length=max_len,
            min_length=min_len,
            length_penalty=2.0,
            num_beams=4,  # 使用 Beam Search 提高品質
            early_stopping=True
        )

        # 解碼摘要
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    except Exception as e:
        print(f"摘要失敗: {e}")
        return text

# 應用摘要函數
df["Explanation"] = df["Explanation"].apply(lambda x: summarize_text(str(x)))

# 存回 CSV
df.to_csv("./knowledge_base/original_data/snopes_mixture_Summarization_T5.csv", index=False)

# try:
#     response = llm.invoke(prompt)
#     return response
# except ValueError as e:
#     if 'max_length' in str(e):
#         return 'format.\n\nClassification: max_token\nConfidence Score: max_token\nJustification: max_token'
#     else:
#         raise e
# except RuntimeError as e:
#     if 'CUDA out of memory' in str(e):
#         torch.cuda.empty_cache()
#         return 'format.\n\nClassification: out_of_memory\nConfidence Score: out_of_memory\nJustification: out_of_memory'
#     else:
#         raise e


# 計算token長度
# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

# text = """"In 2023, the Biden administration tightened the requirements for obtaining an all-access White House press credential called a ""hard pass."" After the stricter requirements went into effect, the number of hard passes issued dropped from 1,417 to 975, in part because many journalists chose not to renew them. However, the Biden administration's changes did not cut or replace reporters in the ""press pool,"" a small group of journalists historically selected by the White House Correspondents' Association to receive privileged access and travel with the president at almost all times. It's unclear how many journalists were made ineligible for hard passes by the Biden administration's tightening of the requirements. The claim was prompted by White House Press Secretary Karoline Leavitt's controversial February 2025 announcement that the Trump administration, not the White House Correspondents' Association, would henceforth select which journalists constitute the press pool. On Feb. 25, 2025, White House Press Secretary Karoline Leavitt announced a consequential change to how the news media covers the U.S. president: The White House Correspondents' Association would no longer be in charge of selecting journalists for the White House press pool, a small group of journalists with privileged access who travel with the president at almost all times. Instead, President Donald Trump's administration would pick the members of the press pool itself. In a statement, WHCA President Eugene Daniels, a reporter for Politico, objected that the White House gave the WHCA no warning and said the decision ""tears at the independence of a free press in the United States."" WHCA board member and Fox News reporter Jacqui Heinrich added on X that the decision would not ""give the power back to the people - it gives power to the White House."" Following those criticisms, Fox News posted an article on Feb. 26 titled ""FLASHBACK: Biden also changed White House press pool, cutting off more than 440 reporters' credentials."" The article likened the Trump administration's press pool change to a decision about journalists' White House access made during the Biden administration. However, comparing the two actions is misleading at best. The Biden administration's change tightened the requirements for obtaining a certain press credential, and, according to Heinrich, presidential administrations have always been in charge of press credentialing. What the Trump administration is trying to change is which small group of White House reporters travel with the president in the ""press pool,"" something a presidential administration has never been in charge of. In 2023, the Biden administration tightened the requirements for the so-called ""hard pass,"" a yearly credential that allows a White House journalist to ""come and go at will"" when the White House is open, according to The Washington Post. That article listed the revised eligibility requirements under Biden as follows: Crucially, one-day press passes still remain available for all journalists, even those who do not meet the above requirements. According to Politico, the number of hard passes dropped from 1,417 to 975 after the new requirements came into effect simply because many journalists chose to not renew the pass — despite the stricter requirements, only one application was denied. A White House spokesperson told Politico that just before the change that ""roughly 40 percent of hard pass holders had not accessed the White House complex in the prior 90 days."" As such, it's misleading to claim that the Biden administration ""cut off"" the credentials of more than 440 reporters — reporters had the chance to renew their credentials, but some decided not to. It is unclear how many of the journalists who did not renew their hard pass credential did so because they wouldn't qualify under the new guidelines (and again, such reporters could still attend news briefings with a day pass). Furthermore, because the general public might misunderstand the term ""press pool"" to mean ""the large group of reporters covering the White House"" instead of the smaller group of journalists that it actually is, it's also misleading to suggest that a change affecting the requirements for press credentials was a change comparable to giving the White House sole control of the makeup of the press pool. The Trump administration's decision did actually affect the small rotating group of journalists who travel with the president — or the people that ""relay the president's activities to the public,"" as The New York Times wrote. The press pool, set by the WHCA, has been traveling with the president since Franklin D. Roosevelt's administration in the 1930s and 1940s, according to the organization's website. The White House's announcement that it would control the press pool itself came not long after the Trump administration barred The Associated Press from the White House for not following its demand to replace the term ""Gulf of Mexico"" with ""Gulf of America."" According to The New York Times, the AP was subsequently kicked out of the pool entirely."""

# tokens = tokenizer(text, return_tensors="pt")["input_ids"]
# print(f"Token 數量: {tokens.shape[1]}")