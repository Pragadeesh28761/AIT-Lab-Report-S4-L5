Pip install openai
import openai
openai.api_key = "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"
completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user",
"content": "Give me 3 ideas that i could build using openai apis"}])
print(completion.choices[0].message.content)

import openai
openai.api_key = "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"
messages = []
system_msg = input("What type of chatbot would you like to create?\n")
messages.append({"role": "system", "content": system_msg})
print("Your new assistant is ready! Type your query")
while input != "quit()":
 message = input()
 messages.append({"role": "user", "content": message})
 response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
 reply = response["choices"][0]["message"]["content"]
 messages.append({"role": "assistant", "content": reply})
 print("\n" + reply + "\n")


import openai
import gradio
openai.api_key = "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"
messages = [{"role": "system", "content": "You are a financial experts that specializes in real
estate investment and negotiation"}]
def CustomChatGPT(user_input):
 messages.append({"role": "user", "content": user_input})
 response = openai.ChatCompletion.create(
 model = "gpt-3.5-turbo",
 messages = messages
 )
 ChatGPT_reply = response["choices"][0]["message"]["content"]
 messages.append({"role": "assistant", "content": ChatGPT_reply})
 return ChatGPT_reply
demo = gradio.Interface(fn=CustomChatGPT, inputs = "text", outputs = "text", title =
"INTELLIGENT CHATBOT")
demo.launch(share=True)
