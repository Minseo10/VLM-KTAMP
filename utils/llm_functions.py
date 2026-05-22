from openai import OpenAI
import json
import requests
import warnings
import numpy as np
import asyncio
import aiohttp
import base64

warnings.filterwarnings('ignore')


def get_chat_completion(
        client,
        chat_history,
        scene_f: str = "prompts/scene_description.txt",
        model: str = "gpt-4-vision-preview",
        max_tokens=2048,
        temperature=0,
        tools=None,
        logprobs=None,
        top_logprobs=None,
) -> str:
    params = {
        "model": model,
        "messages": chat_history,
        "max_tokens": max_tokens,
        "temperature": temperature,
        # "stop": stop,
        "logprobs": logprobs,
        "top_logprobs": top_logprobs,
    }
    if tools:
        params["tools"] = tools

    # read updated scene description
    with open(scene_f, "r") as f:
        scene = f.read()

    # append scene description
    chat_history.append(
        {
            "role": "user",
            "content": scene,
        }
    )

    completion = client.chat.completions.create(**params)

    chat_history.append(
        {
            "role": "assistant",
            "content": completion.choices[0].message.content,
        }
    )

    return completion


# asks user's question to chatgpt & update scene initialization
def ask(prompt, model, client, chat_history, scene_f):
    chat_history.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    # read updated scene description
    with open(scene_f, "r") as f:
        scene = f.read()

    # append scene description
    chat_history.append(
        {
            "role": "user",
            "content": scene,
        }
    )

    # completion = client.chat.completions.create(
    #     model="gpt-4-vision-preview",
    #     messages=chat_history,
    #     max_tokens=2048,  # default value is very low, so you have to increase it
    #     temperature=0.0
    # )

    completion = get_chat_completion(client, chat_history, model=model)

    chat_history.append(
        {
            "role": "assistant",
            "content": completion.choices[0].message.content,
        }
    )
    # print(completion.choices[0].message.content)
    return completion.choices[0].message.content


def get_session_completion(chat_history, K, model="gpt-4o", max_tokens=4095,
                           temperature=0.7, logprobs=True, top_logprobs=None):
    with open("../config.json", "r") as f:
        config = json.load(f)
    api_key = config["OPENAI_API_KEY"]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # messages = [
    #     {
    #         "role": "system",
    #         "content": chat_history[0]['content']
    #     },
    #     {
    #         "role": "user",
    #         "content": prompt
    #     }
    # ]

    def fetch_completion(messages):
        resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers,
                             json={
                                 "model": model,
                                 "messages": messages,
                                 "max_tokens": max_tokens,
                                 "temperature": temperature,
                                 "logprobs": logprobs,
                                 "top_logprobs": top_logprobs,
                             })

        return resp.json()

    result = fetch_completion(chat_history)

    if result['choices'][0]['finish_reason'] == 'length':
        chat_history.append({
            "role": "assistant",
            "content": result["choices"][0]['message']["content"]
        })
        chat_history.append({
            "role": "user",
            "content": f'''Continue.'''
        })
        next_result = fetch_completion(chat_history)

        result["choices"][0]['message']["content"] += "\n"
        result["choices"][0]['logprobs']["content"].append({'token': '\n', 'logprob': -3.0e-05, 'bytes': [10], 'top_logprobs': []})
        result["choices"][0]['message']["content"] += next_result["choices"][0]['message']["content"]
        result["choices"][0]['logprobs']["content"] += next_result["choices"][0]['logprobs']["content"]

    return result

# Function to encode the image
def encode_image(image_path_list):
    image_list = []
    for image_path in image_path_list:
        with open(image_path, "rb") as image_file:
            image =  base64.b64encode(image_file.read()).decode("utf-8")
            image_list.append(image)
    return image_list

def create_file(client, file_path):
  with open(file_path, "rb") as file_content:
    result = client.files.create(
        file=file_content,
        purpose="vision",
    )
    return result.id

def ask_multiple_image(images_path, prompt, model="gpt-4o", max_tokens=4095, temperature=0.7):
    base64_images = []
    for image_path in images_path:
        base64_image = encode_image(image_path)
        base64_images.append(base64_image)

    with open("../config.json", "r") as f:
        config = json.load(f)
    api_key = config["OPENAI_API_KEY"]
    client = OpenAI(api_key=api_key)

    message_content = []
    message_content.append({
        "type": "text",
        "text": prompt
    })
    for base64_image in base64_images:
        message_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        })

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": message_content
            }
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    # print("message_content:", message_content)
    # print("reponse: ", response.choices[0])
    return response.choices[0]

def seq_prob(completion):
    if 'choices' not in completion:
        print("Invalid response structure:", json.dumps(completion, indent=2))
        raise KeyError("The completion response does not contain 'choices' key")
    response_text = completion["choices"][0]['message']["content"].strip()
    logprobs_content = completion["choices"][0]['logprobs']['content']
    token_logprobs = [token['logprob'] for token in logprobs_content]
    tokens = [token['token'] for token in logprobs_content]

    action_logprobs = []
    current_action_logprob = 0
    action_lines = []

    # Split response text into lines
    lines = response_text.split('\n')

    # Filter out comment lines and collect non-comment lines
    for line in lines:
        line = line.strip()
        if not line or line.startswith(';'):
            continue
        action_lines.append(line)

    # Process tokens to calculate action log probabilities
    action_idx = 0
    for token, logprob in zip(tokens, token_logprobs):
        if token.endswith(',') or token.endswith('\n'):
            current_action_logprob += logprob
            if action_idx < len(action_lines):
                action_logprobs.append(current_action_logprob)
                current_action_logprob = 0
                action_idx += 1
        else:
            current_action_logprob += logprob
    if current_action_logprob != 0 and action_idx < len(action_lines):
        action_logprobs.append(current_action_logprob)

    action_probabilities = [np.exp(logprob) for logprob in action_logprobs]
    # actions = response_text.split('\n')
    # actions = [action.strip() for action in actions if action.strip()]
    # actions = [action.split(',') for action in actions]

    # Ensure we do not exceed the number of actions found
    if len(action_probabilities) > len(action_lines):
        action_probabilities = action_probabilities[:len(action_lines)]

    actions_with_probabilities = list(zip(action_lines, action_probabilities))

    # filtered_actions_with_probabilities = [
    #     (action, prob) for action, prob in actions_with_probabilities
    #     if action.startswith('(') and action.endswith(')')
    # ]

    return actions_with_probabilities


async def get_async_completion(prompt_list, chat_history, K, max_parallel_calls=10, model="gpt-4o",
                               max_tokens=4095, temperature=0.0, logprobs=True, top_logprobs=None, timeout=30,
                               max_retries=3, retry_delay=4, parallel=True):
    with open("../config.json", "r") as f:
        config = json.load(f)
    api_key = config["OPENAI_API_KEY"]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    if parallel:
        semaphore = asyncio.Semaphore(value=max_parallel_calls)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(timeout)) as session:
        async def fetch_completion(prompt, retry_count=0):
            if parallel:
                async with semaphore:
                    async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json={
                        "model": model,
                        "messages": [
                            {
                                "role": "system",
                                "content": chat_history[0]['content']
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "logprobs": logprobs,
                        "top_logprobs": top_logprobs,
                    }) as resp:
                        try:
                            result = await resp.json()
                            if 'choices' not in result:
                                raise KeyError("The completion response does not contain 'choices' key")
                            return result
                        except KeyError as e:
                            if retry_count < max_retries:
                                print(f"Error fetching completion: {e}. Retrying in {retry_delay} seconds.")
                                await asyncio.sleep(retry_delay)
                                return await fetch_completion(prompt, retry_count + 1)
                            else:
                                raise
            else:
                async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json={
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": chat_history[0]['content']
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "logprobs": logprobs,
                    "top_logprobs": top_logprobs,
                }) as resp:
                    try:
                        result = await resp.json()
                        if 'choices' not in result:
                            raise KeyError("The completion response does not contain 'choices' key")
                        return result
                    except KeyError as e:
                        if retry_count < max_retries:
                            print(f"Error fetching completion: {e}. Retrying in {retry_delay} seconds.")
                            await asyncio.sleep(retry_delay)
                            return await fetch_completion(prompt, retry_count + 1)
                        else:
                            raise

        if parallel:
            # Limit the prompt_list to the first K items
            prompt_list = prompt_list[:K]
            return await asyncio.gather(*[fetch_completion(prompt) for prompt in prompt_list])
        else:
            results = []
            for prompt in prompt_list[:K]:
                results.append(await fetch_completion(prompt))
            return results
