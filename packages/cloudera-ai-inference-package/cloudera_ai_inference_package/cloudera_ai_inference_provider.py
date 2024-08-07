import base64
import json
import logging
import os
import requests
import subprocess
from typing import Any, AsyncIterator, Dict, Iterator, List, Mapping, Optional

from jupyter_ai_magics import BaseProvider
from jupyter_ai_magics.providers import MultiEnvAuthStrategy
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import SimpleChatModel
from langchain.llms.base import LLM
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage, BaseMessageChunk, HumanMessage, SystemMessage
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.runnables.config import RunnableConfig

from cloudera_ai_inference_package.model_discovery import getCopilotModels


class ClouderaAIInferenceLanguageModelProvider(BaseProvider, SimpleChatModel, LLM):
    id = "cloudera"
    name = "Cloudera AI Inference Provider"
    model = ""
    model_id_key = ""

    copilot_config_dir = os.getenv("COPILOT_CONFIG_DIR") or ""
    ai_inference_models, models = getCopilotModels(copilot_config_dir, model_type="inference")
    cdp_cli_path = "/Users/gavan/Library/Python/3.9/bin/cdp"
    # "/home/cdsw/.local/bin/cdp"

    MultiEnvAuthStrategy(names=["CDP_PRIVATE_KEY", "CDP_ACCESS_KEY_ID", "CDP_REGION", "CDP_ENDPOINT_URL", "ENDPOINT_URL"])

    def WriteParamsToCDPConfig(self, cdp_private_key, cdp_access_key_id, cdp_region, endpoint_url, cdp_endpoint_url):
        logging.error("Writing params to CDP config.")
        if not os.path.exists(self.cdp_cli_path):
            return False
        completed_process = subprocess.run([self.cdp_cli_path, "configure", "set", "cdp_private_key", cdp_private_key])
        if completed_process.returncode != 0:
            return False
        completed_process = subprocess.run([self.cdp_cli_path, "configure", "set", "cdp_access_key_id", cdp_access_key_id])
        if completed_process.returncode != 0:
            return False
        completed_process = subprocess.run([self.cdp_cli_path, "configure", "set", "cdp_region", cdp_region])
        if completed_process.returncode != 0:
            return False
        completed_process = subprocess.run([self.cdp_cli_path, "configure", "set", "endpoint_url", endpoint_url])
        if completed_process.returncode != 0:
            return False

        completed_process = subprocess.run([self.cdp_cli_path, "configure", "set", "cdp_endpoint_url", cdp_endpoint_url])
        return completed_process.returncode == 0

    def GetCDPToken(self):
        if not os.path.exists(self.cdp_cli_path):
            logging.error("CDP CLI is not installed. Please install by running the command 'pip install cdpcli' in Terminal Access.")
            return None
        logging.error("Found CDP CLI")

        completed_process = subprocess.run([self.cdp_cli_path, "iam", "generate-workload-auth-token", "--workload-name", "DE"], capture_output = True, text = True)
        try:
            completed_process.check_returncode()
        except:
            logging.error("Unable to get CDP Token.")
            return None

        response = json.loads(completed_process.stdout)
        return response['token']

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = kwargs.get("model_id")
        logging.basicConfig(filename="copilot.txt", level=logging.DEBUG, format="")
        cdp_private_key = os.getenv("CDP_PRIVATE_KEY")
        cdp_access_key_id = os.getenv("CDP_ACCESS_KEY_ID")
        cdp_region = os.getenv("CDP_REGION")
        endpoint_url = os.getenv("ENDPOINT_URL")
        cdp_endpoint_url = os.getenv("CDP_ENDPOINT_URL")
        cdp_cli_path = self.cdp_cli_path
        if not (cdp_private_key and cdp_access_key_id and cdp_region and cdp_endpoint_url and endpoint_url and cdp_cli_path):
            return
        self.WriteParamsToCDPConfig(cdp_private_key, cdp_access_key_id, cdp_region, endpoint_url, cdp_endpoint_url)

    @property
    def _llm_type(self) -> str:
        return "cloudera"

    def GetInferenceEndpoint(self, model):
        logging.error(self.ai_inference_models)
        for endpoint in self.ai_inference_models:
            logging.error(endpoint)
            if endpoint['name'] == model:
                return endpoint['endpoint']
        return ""

    def BuildChatCompletionMessage(self, messages):
        request_messages = []
        for message in messages:
            role = "assistant"
            if isinstance(message, SystemMessage):
                role = "system"
            elif isinstance(message, HumanMessage):
                role = "user"
            request_messages.append({"role": role, "content": message.content})
        return request_messages

    def BuildCompletionPrompt(self, messages):
        prompt = ""
        for message in messages:
            if isinstance(message, SystemMessage):
                prompt += f"{message.content}\n"
            elif isinstance(message, HumanMessage):
                prompt += f"\nHuman: {message.content}"
            else:
                prompt += f"\nAI: {message.content}"
        prompt += "\nAI: "
        return prompt

    def _stream(
        # Support streaming output tokens.
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any
    ) -> Iterator[ChatGenerationChunk]:
        inference_endpoint = self.GetInferenceEndpoint(self.model)
        if not inference_endpoint:
            logging.error("Unable to find endpoint: " + self.model)
            return "Error: unable to find endpoint: " + self.model

        cdp_token = self.GetCDPToken()
        if not cdp_token:
            logging.error("Unable to get CDP Token.")
            return "Unable to get CDP Token. Please install CDP CLI by running the command 'pip install cdpcli' in Terminal Access and make sure all required environment variables are set. See https://docs.cloudera.com/r/jupyter-copilot for more details."

        if inference_endpoint.find("chat/completions") != -1:
            # OpenAI Chat completions API
            request_messages = self.BuildChatCompletionMessage(messages)

            request = {"messages": request_messages, "model": self.model, "temperature": 1, "max_tokens": 256, "stream": True}
            logging.info(f"request: {request}")
            try:
                r = requests.post(
                    inference_endpoint,
                    data=json.dumps(request),
                    headers={'Content-Type': 'application/json',
                    'Authorization': 'Bearer ' + cdp_token},
                    verify=True,
                    stream=True
                )
            except Exception as e:
                logging.error(e)
                return "Request to Cloudera AI Inference Service failed."

            for line in r.iter_lines():
                decoded_line = line.decode('utf-8').removeprefix("data: ")
                if decoded_line == "":
                    continue
                if decoded_line == "[DONE]":
                    break
                line_json = json.loads(decoded_line)
                if "choices" in line_json and "delta" in line_json["choices"][0] and "content" in line_json["choices"][0]["delta"]:
                    yield ChatGenerationChunk(message=AIMessageChunk(content=line_json["choices"][0]["delta"]["content"]))
        elif inference_endpoint.find("completions") != -1:
            # OpenAI Completions API
            prompt = self.BuildCompletionPrompt(messages)
            req_data = '{"prompt": "' + prompt.encode('unicode_escape').decode("utf-8")

            my_req_data = req_data + '","model":"' + self.model + '","temperature":1,"max_tokens":256,"stream":true}'
            logging.info('req:')
            logging.info(my_req_data)

            try:
                r = requests.post(
                    inference_endpoint,
                    data = my_req_data,
                    headers={'Content-Type': 'application/json',
                    'Authorization': 'Bearer ' + cdp_token},
                    verify=True,
                    stream=True
                )
            except Exception as e:
                logging.error(e)
                return "Request to Cloudera AI Inference Service failed."

            for line in r.iter_lines():
                decoded_line = line.decode('utf-8').removeprefix("data: ")
                if decoded_line == "":
                    continue
                if decoded_line == "[DONE]":
                    break
                line_json = json.loads(decoded_line)
                if "choices" in line_json and "text" in line_json["choices"][0]:
                    yield ChatGenerationChunk(message=AIMessageChunk(content=line_json["choices"][0]["text"]))

    def _call(
        # Supports notebook generation and magic commands.
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        inference_endpoint = self.GetInferenceEndpoint(self.model)
        if not inference_endpoint:
            logging.error("Unable to find endpoint: " + self.model)
            return "Error: unable to find endpoint: " + self.model

        cdp_token = self.GetCDPToken()
        if not cdp_token:
            logging.error("Unable to get CDP Token.")
            return "Unable to get CDP Token. Please install CDP CLI by running the command 'pip install cdpcli' in Terminal Access and make sure all required environment variables are set. See https://docs.cloudera.com/r/jupyter-copilot for more details."

        if inference_endpoint.find("chat/completions") != -1:
            # OpenAI Chat completions API
            request_messages = self.BuildChatCompletionMessage(messages)
            request = {"messages": request_messages, "model": self.model, "temperature": 1, "max_tokens": 1024, "stream": False}
            logging.info(json.dumps(request))
            try:
                r = requests.post(inference_endpoint,
                                  data=json.dumps(request),
                                  headers={'Content-Type': 'application/json',
                                  'Authorization': 'Bearer ' + cdp_token},
                                  verify=True,
                                  stream=False)
                rjson = r.json()
                logging.info(f"rjson: {rjson}")
                if rjson and rjson['choices'] and len(rjson['choices']) > 0 and rjson['choices'][0]['message'] and rjson['choices'][0]['message']['content']:
                    return rjson['choices'][0]['message']['content']
            except Exception as e:
                logging.error(e)
                return "Request to Cloudera AI Inference Service failed."
        elif inference_endpoint.find("completions") != -1:
            # OpenAI Completions API
            prompt = self.BuildCompletionPrompt(messages)
            logging.info(f"prompt: {prompt}")
            request = {"prompt": prompt, "model": self.model, "temperature": 1, "max_tokens": 1024, "stream": False}
            logging.info(json.dumps(request))

            try:
                r = requests.post(
                    inference_endpoint,
                    data = json.dumps(request),
                    headers={'Content-Type': 'application/json',
                    'Authorization': 'Bearer ' + cdp_token},
                    verify=True,
                    stream=False
                )
                rjson = r.json()
                logging.info(f"rjson: {rjson}")
                if r.json() and rjson['choices'] and len(rjson['choices']) > 0 and rjson['choices'][0]['text']:
                    return rjson['choices'][0]['text']
            except Exception as e:
                logging.error(e)
                return "Request to Cloudera AI Inference Service failed."

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        # Get the identifying parameters.
        return {}
