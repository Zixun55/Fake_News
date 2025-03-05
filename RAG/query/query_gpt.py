import openai

# 設定 OpenAI API 金鑰


try:
    # 使用新版 API 呼叫
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Test access to GPT-3.5-turbo model."}
        ],
        max_tokens=50
    )
    print("測試成功！模型回應：", response["choices"][0]["message"]["content"])
except openai.error.OpenAIError as e:  # 錯誤處理模組名稱修正
    print("測試失敗，錯誤訊息：", str(e))