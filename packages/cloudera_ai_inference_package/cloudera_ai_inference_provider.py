from typing import Any, AsyncIterator, Dict, Iterator, List, Mapping, Optional
import json
import logging
import os
import base64
import subprocess

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage, BaseMessageChunk
from langchain_core.messages.ai import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.runnables.config import RunnableConfig

def stringToB64String(s):
    return base64.b64encode(s.encode('utf-8')).decode("utf-8")

def b64StringToString(b):
    return base64.b64decode(b).decode('utf-8')

from jupyter_ai_magics import BaseProvider
from jupyter_ai_magics.providers import MultiEnvAuthStrategy
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import SimpleChatModel
from langchain.llms.base import LLM
import requests

def getCopilotModels(config_dir):
    if not config_dir or not os.path.exists(config_dir):
        return [], []

    f = open(config_dir)
    copilot_config = json.load(f)
    ai_inference_models = []
    if copilot_config and "aiInferenceModels" in copilot_config and copilot_config["aiInferenceModels"]:
        ai_inference_models = copilot_config['aiInferenceModels']
    f.close()

    models = []

    for ai_inference_model in ai_inference_models:
        models.append(ai_inference_model['name'])

    return ai_inference_models, models

class ClouderaAIInferenceProvider(BaseProvider, SimpleChatModel, LLM):
    api_key = "API_KEY2"
    id = "cloudera"
    name = "Cloudera AI Inference Provider"
    model_id_key = "vllm_llama2"
    model = ""

    copilot_config_dir = os.getenv("COPILOT_CONFIG_DIR") or ""
    ai_inference_models, models = getCopilotModels(copilot_config_dir)
    cdp_cli_path = "/home/cdsw/.local/bin/cdp"

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
        model = kwargs.get("model_id")
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


    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any) -> Iterator[ChatGenerationChunk]:

        inference_endpoint = self.GetInferenceEndpoint(self.model)
        if not inference_endpoint:
            logging.error("Unable to find endpoint: " + self.model)
            return "Error: unable to find endpoint: " + self.model
        req_data = '{"prompt": "' + messages[-1].content.encode('unicode_escape').decode("utf-8")

        my_req_data = req_data + '","model":"' + self.model + '","temperature":1,"max_tokens":256,"stream":true}'
        logging.info('req:')
        logging.info(my_req_data)

        cdp_token = self.GetCDPToken()
        if not cdp_token:
            logging.error("Unable to get CDP Token.")
            return "Unable to get CDP Token. Please install CDP CLI by running the command 'pip install cdpcli' in Terminal Access and make sure all required environment variables are set. See https://docs.cloudera.com/r/jupyter-copilot for more details."
        try:
            r = requests.post(inference_endpoint,
                              data = my_req_data,
                              headers={'Content-Type': 'application/json',
                              'Authorization': 'Bearer ' + cdp_token},
                              verify=True,
                              stream=True)
        except Exception as e:
            logging.error(e)
            return "Request to Cloudera AI Inference Service failed."

        for line in r.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line and len(decoded_line) > 6:
                    data = decoded_line[6:]
                    if data != "[DONE]":
                        line_json = json.loads(decoded_line[6:])
                        if "choices" in line_json:
                            if "text" in line_json["choices"][0]:
                                chunk = ChatGenerationChunk(message=AIMessageChunk(content=line_json["choices"][0]["text"]))
                                yield chunk

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        logging.error("_call")
        logging.error(prompt)
        return prompt

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"api_key": self.api_key}
