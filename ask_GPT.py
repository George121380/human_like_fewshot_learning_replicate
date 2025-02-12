from openai import OpenAI
import time

# LLM_MODEL = "gpt-4o"
LLM_MODEL = "deepseek"

def ask_GPT(system,content):
    while True:
        try:
            if LLM_MODEL == "gpt-4o": # GPT-4o api
                with open("/Users/liupeiqi/workshop/Research/api_key.txt","r") as f:
                    api_key = f.read().strip()
                client = OpenAI(api_key=api_key)
                completion = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content":content}
                    ]
                )
            elif LLM_MODEL == "deepseek": # Deepseek api
                client = OpenAI(api_key="sk-7eb58550af8a4042aca7d33d495ec2e0", base_url="https://api.deepseek.com")
                completion = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": content},
                    ],
                    stream=False
                )
            else:
                raise ValueError("Invalid LLM_MODEL")
            
            return completion.choices[0].message.content
            
        except Exception as e:
            print(e)
            time.sleep(1)
            continue