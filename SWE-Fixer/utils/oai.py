from openai import OpenAI
from tqdm import tqdm

import queue
import threading


def count_tokens(tokenizer, text):
    if tokenizer is None:
        raise ValueError("Tokenizer has not been initialised")

    if hasattr(tokenizer, "encode"):
        return len(tokenizer.encode(text))

    if callable(tokenizer):
        encoded = tokenizer(text)
        if isinstance(encoded, dict) and "input_ids" in encoded:
            return len(encoded["input_ids"])

    raise TypeError(
        "Unsupported tokenizer object: expected tiktoken encoding or callable"
    )


def load_openai_client(api_key, base_url):
    return OpenAI(api_key=api_key, base_url=base_url)


def get_model_prediction(client, input_text, system_prompt=None, timeout=300, temp=0):
    def make_request(result_queue):
        if temp != 0:
            print("temp: ", temp)
        try:
            model_name = client.models.list().data[0].id
            if system_prompt is None or system_prompt == "":
                message = [
                    {"role": "user", "content": input_text},
                ]
            else:
                message = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_text},
                ]
            response = client.chat.completions.create(
                model=model_name,
                messages=message,
                temperature=temp,
                top_p=1,
            )
            result_queue.put(response.choices[0].message.content)
        except Exception as e:
            result_queue.put(str(e))

    result_queue = queue.Queue()
    request_thread = threading.Thread(target=make_request, args=(result_queue,))
    request_thread.start()
    request_thread.join(timeout=timeout)

    if request_thread.is_alive():
        return ""

    return result_queue.get()
