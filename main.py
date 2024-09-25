from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import requests
import re

app = FastAPI()

client = InferenceClient(model="HuggingFaceH4/zephyr-7b-beta", token="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")

class Message(BaseModel):
    content: str

class ChatRequest(BaseModel):
    message: Message
    history: list[Message] = []

@app.post("/chat/")
def chat_endpoint(request: ChatRequest):
    try:
        # System message for the chatbot
        system_message = {"role": "system", "content": "You are a friendly Chatbot."}

        # Translate the user's message to English
        translated_message = translate_text("uz", "en", request.message.content)

        # Prepare the messages for the chatbot
        messages = [system_message] + [{"role": "user", "content": m.content} for m in request.history] + [{"role": "user", "content": translated_message}]

        # Get the chatbot's response
        response = ""
        for message_part in client.chat_completion(messages=messages, max_tokens=512, stream=True, temperature=0.7, top_p=0.95):
            token = message_part["choices"][0]["delta"].get("content", "")
            response += token

        # Translate the chatbot's response back to Uzbek
        translated_response = preserve_code_and_translate("en", "uz", response)

        # Update history with the new messages
        updated_history = request.history + [Message(content=request.message.content), Message(content=translated_response)]

        return {"response": translated_response, "history": updated_history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def translate_text(from_lang: str, to_lang: str, input_text: str) -> str:
    url = "https://translate.googleapis.com/translate_a/single"
    translated_text = ""

    text_chunks = re.split(r'(?<=[.!?]) +', input_text)

    for chunk in text_chunks:
        params = {
            "client": "gtx",
            "sl": from_lang,
            "tl": to_lang,
            "dt": "t",
            "q": chunk
        }

        try:
            res = requests.get(url, params=params)
            res.raise_for_status()
            translated_chunk = res.json()[0][0][0]
            translated_text += translated_chunk + " "

        except requests.exceptions.RequestException as e:
            raise Exception(f"An error occurred: {str(e)}")

    return translated_text.strip()

def preserve_code_and_translate(from_lang: str, to_lang: str, text: str) -> str:
    code_pattern = re.compile(r"(```.*?```|`[^`]*`)", re.DOTALL)

    parts = []
    last_end = 0

    for match in code_pattern.finditer(text):
        non_code_text = text[last_end:match.start()]
        translated_text = translate_text(from_lang, to_lang, non_code_text)
        parts.append(translated_text)
        parts.append(match.group(0))
        last_end = match.end()

    remaining_text = text[last_end:]
    translated_remaining = translate_text(from_lang, to_lang, remaining_text)
    parts.append(translated_remaining)

    return ''.join(parts)
