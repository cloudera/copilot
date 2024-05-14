from typing import Any, List, Mapping, Optional
import json
import logging
import os
import base64
import subprocess

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
    logging.error(copilot_config)
    ai_inference_models = []
    if copilot_config and "aiInferenceModels" in copilot_config:
        ai_inference_models = copilot_config['aiInferenceModels']
    f.close()

    models = []

    for ai_inference_model in ai_inference_models:
        logging.error(ai_inference_model)
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
        logging.error("CDP CLI Path:")
        logging.error(self.cdp_cli_path)
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

        logging.error("CDPCLI response: ")
        logging.error(completed_process.stdout)
        response = json.loads(completed_process.stdout)
        logging.error("Parsed CDPCLI response: ")
        logging.error(response)
        return response['token']

    def __init__(self, **kwargs):
        print("init kwargs:", kwargs)
        super().__init__(**kwargs)
        self.model = kwargs.get("model_id")
        logging.basicConfig(filename="copilot.txt", level=logging.DEBUG, format="")
        model = kwargs.get("model_id")
        '''cdp_private_key = os.getenv("CDP_PRIVATE_KEY")
        cdp_access_key_id = os.getenv("CDP_ACCESS_KEY_ID")
        cdp_region = os.getenv("CDP_REGION")
        endpoint_url = os.getenv("ENDPOINT_URL")
        cdp_endpoint_url = os.getenv("CDP_ENDPOINT_URL")
        cdp_cli_path = self.cdp_cli_path
        if not (cdp_private_key and cdp_access_key_id and cdp_region and cdp_endpoint_url and endpoint_url and cdp_cli_path):
            return
        self.WriteParamsToCDPConfig(cdp_private_key, cdp_access_key_id, cdp_region, endpoint_url, cdp_endpoint_url)'''

    @property
    def _llm_type(self) -> str:
        return "custom1"

    def GetInferenceEndpoint(self, model):
        logging.error(self.ai_inference_models)
        for endpoint in self.ai_inference_models:
            logging.error(endpoint)
            if endpoint['name'] == model:
                return endpoint['endpoint']
        return ""

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        inference_endpoint = self.GetInferenceEndpoint(self.model)
        if not inference_endpoint:
            logging.error("Unable to find endpoint: " + self.model)
            return "Error: unable to find endpoint: " + self.model

        logging.error("call kwargs:")
        logging.error(kwargs)
        logging.error('prompt:')
        logging.error(prompt)
        logging.error('prompt:')
        logging.error(prompt[-1].content)
        req_data = '{"prompt":"[INST]<<SYS>>You are Cloudera Copilot, a conversational assistant living in JupyterLab to help users. You are not a language model, but rather an application built on a foundation model from Cloudera AI Inference Provider called casperhansen_llama-3-8b-instruct-awq. You are talkative and you provide lots of specific details from the foundation model\'s context. You may use Markdown to format your response. Code blocks must be formatted in Markdown. Math should be rendered with inline TeX markup, surrounded by $. If you do not know the answer to a question, answer truthfully by responding that you do not know. The following is a friendly conversation between you and a human.<</SYS>> ' + prompt[-1].content.encode('unicode_escape').decode("utf-8") + '[/INST]'

        my_req_data = req_data + '","model":"' + self.model + '","temperature":1,"max_tokens":256}'
        logging.error('req:')
        logging.error(my_req_data)
        logging.error('req:')
        logging.error(my_req_data)

        cdp_token = self.GetCDPToken()
        logging.error('cdp_token:')
        logging.error(cdp_token)
        if not cdp_token:
            logging.error("Unable to get CDP Token.")
            return "Unable to get CDP Token. Please install CDP CLI by running the command 'pip install cdpcli' in Terminal Access and make sure all required environment variables are set. See https://docs.cloudera.com/r/jupyter-copilot for more details."
        logging.error('cdp_token:')
        logging.error(cdp_token)
        try:
            r = requests.post(inference_endpoint,
                              data = my_req_data,
                              headers={'Content-Type': 'application/json',
                              'Authorization': 'Bearer ' + cdp_token}, verify=False)
        except:
            return "Request to Cloudera AI Inference Service failed."

        logging.error('r:')
        logging.error(r)
        logging.error('rjson:')
        logging.error(r.json())
        if r.json()['choices'] and len(r.json()['choices']) > 0 and r.json()['choices'][0]['text']:
            return r.json()['choices'][0]['text']
        logging.error('response:')
        return r.json()

        decoded_str = b64StringToString(r)
        logging.error('rjson:')
        logging.error(decoded_str.json())
        if 'errors' in decoded_str.json():
            logging.error('error:')
            logging.error(decoded_str.json()['errors'])
            return decoded_str.json()['errors']
        elif 'predictions' in decoded_str.json():
            logging.error('predictions:')
            logging.error(decoded_str.json()['predictions'])
            return decoded_str.json()['predictions'][0]
        else:
            logging.error('response:')
            logging.error(decoded_str.json()['response'])
            return decoded_str.json()['response']

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"api_key": self.api_key}
