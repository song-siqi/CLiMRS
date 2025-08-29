import requests
import json
import sys
from .args import *

MODEL_DEFAULT = "gpt-4o-2024-11-20"

class Agent:
    '''
        This is a class for querying the LLM model as an agent.
        :param str model: the model name to use.
        :param str api_url: str, the API URL to use for the model.
        :param str api_key: the API key to use for authentication.
    '''
    def __init__(
            self,
            model=None,
            api_url=None,
            api_key=None,

        ):

        self.model_name = model
        self.api_url = api_url
        self.api_key = api_key

        self.headers  = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {api_key}',
            'User-Agent':'Apifox/1.0.0(https://apifox.com)',
            'Content-Type': 'application/json'
        }

        self.api_url_completions = f"{self.api_url}/v1/chat/completions"
        pass

    def respond_once_raw(
            self,
            messages=None,
            max_tokens=None,
            temperature=None,
        ):

        assert messages is not None, "Messages cannot be None!"

        payload_dict = {
            "model": self.model_name,
            "messages": messages
        }

        if max_tokens is not None:
            payload_dict["max_tokens"] = max_tokens
        if temperature is not None:
            payload_dict["temperature"] = temperature

        url = self.api_url_completions
        payload = json.dumps(payload_dict)
        response = requests.post(url, headers=self.headers, data=payload)
        response = response.json()

        return response
    
    def respond_once_all_args(
            self,
            messages=None,
            *args,
            **kwargs
        ):

        assert messages is not None, "Messages cannot be None!"

        payload_dict = {
            "model": self.model_name,
            "messages": messages
        }

        payload_dict.update(kwargs)

        url = self.api_url_completions
        payload = json.dumps(payload_dict)
        response = requests.post(url, headers=self.headers, data=payload)

        if response.status_code != 200:
            print(f"Request failed with status code {response.status_code}")
            print(response.text)  # 可能是 HTML 错误页或者空字符串
            response.raise_for_status()

        # 检查内容是否非空再尝试解析
        if response.text.strip():
            try:
                json_data = response.json()
            except json.JSONDecodeError as e:
                print("Failed to parse JSON:", e)
                print("Response text:", response.text)
        else:
            print("Empty response body")
        # import pdb; pdb.set_trace()
        
        response = response.json()

        return response
    
    def respond_once(
            self,
            question=None,
        ):
        messages = self.prompt_from_question(question)
        return self.respond_once_raw(messages)

    def prompt_from_question(
            self,
            question
        ):
        messages = [
            {"role": "user", "content": question}
        ]
        return messages


    def usage_from_response(
            self,
            response
        ):
        usage = response.get("usage", {})
        return usage
    
    def answer_from_response(
            self,
            response
        ):
        answer = response.get("choices", [])[0].get("message", {}).get("content", "")
        return answer

 
if __name__ == "__main__":
    agent = Agent(
        model=MODEL_DEFAULT,
        api_url=API_URL,
        api_key=API_KEY_SIQI,
    )

    '''
    
    response = agent.respond_once_raw(
        messages=[
            {
                "role": "system",
                "content": "You are a Computer Science researcher specialized in Reinforcement Learning, Robotics, \
                            and Generative Models. Please help me with the following questions: "
            },
            {
                "role": "user",
                "content": "What is the best way to train a reinforcement learning agent?"
            }
        ],
        max_tokens=512,
        temperature=0,
    )

    # print(response)
    answer = agent.answer_from_response(response)
    usage = agent.usage_from_response(response)

    print(f"Answer: {answer}\n")
    print(f"Usage: {usage}\n")

    import pdb; pdb.set_trace()
    
    '''
