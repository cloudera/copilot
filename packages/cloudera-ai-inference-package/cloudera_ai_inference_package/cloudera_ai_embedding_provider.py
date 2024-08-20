import json
import logging
import os

import requests
from jupyter_ai_magics import BaseEmbeddingsProvider
from langchain_core.embeddings import Embeddings

from cloudera_ai_inference_package.auth import getAccessToken
from cloudera_ai_inference_package.error_messages import CopilotErrorMessages
from cloudera_ai_inference_package.model_discovery import getCopilotModels


class ClouderaAIInferenceEmbeddingModelProvider(BaseEmbeddingsProvider, Embeddings):
    id = "cloudera"
    name = "Cloudera AI Embedding Provider"
    model_id_key = "model_id"
    jwt_path = '/tmp/jwt'

    API_SYNTAX = "OPENAI"
    ai_inference_models, models = getCopilotModels(
        os.getenv("COPILOT_CONFIG_DIR", ""), model_type="embedding"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_endpoint = self._get_inference_endpoint()

    def _get_inference_endpoint(self):
        for ai_inference_model in self.ai_inference_models:
            if ai_inference_model["name"] == self.model_id:
                return ai_inference_model["endpoint"]
        return ""

    def embed_documents(self, texts):
        return self._call(texts, "passage")

    def embed_query(self, text):
        return self._call([text], "query")[0]

    def _call(self, texts, mode):
        if not self.model_endpoint:
            raise ValueError(
                CopilotErrorMessages.MODEL_NOT_CONFIGURED, {"model": self.model_id}
            )
        if mode not in ["passage", "query"]:
            raise ValueError(
                CopilotErrorMessages.INTERNAL_ERROR, {"unknown_mode": mode}
            )

        # https://developer.nvidia.com/docs/nemo-microservices/embedding/source/openai-api.html#api-spec
        if self.API_SYNTAX == "OPENAI":
            payload = json.dumps({"input": texts, "model": self.model_id + "-" + mode})

        elif self.API_SYNTAX == "NIM":
            payload = json.dumps(
                {"input": texts, "input_type": mode, "model": self.model_id}
            )
        else:
            raise ValueError(
                CopilotErrorMessages.INTERNAL_ERROR, {"API_SYNTAX": self.API_SYNTAX}
            )

        access_token = getAccessToken(self.jwt_path)
        headers = {"Content-Type": "application/json",
                   "Bearer": access_token}
        try:
            response = requests.post(self.model_endpoint, headers=headers, data=payload)
        except Exception as e:
            raise RuntimeError(
                CopilotErrorMessages.MODEL_RESPONSE_ERROR,
                {"reason": "failed talking to the model endpoint"},
            )
        if not response.ok:
            raise RuntimeError(
                CopilotErrorMessages.MODEL_RESPONSE_ERROR,
                {
                    "reason": "not_ok_status",
                    "response_http_status": response.status_code,
                    "response": response.text,
                    "endpoint": self.model_endpoint,
                },
            )
        try:
            response_json = response.json()
        except Exception as e:
            raise RuntimeError(
                CopilotErrorMessages.MODEL_RESPONSE_ERROR,
                {"reason": "not a JSON response"},
            )
        if not "data" in response_json:
            raise KeyError(
                CopilotErrorMessages.MODEL_RESPONSE_ERROR,
                {"reason": "no data field in response"},
            )

        embeddings = self._extract_embedding_from_api_response(response_json["data"])
        if len(embeddings) != len(texts):
            raise RuntimeError(
                CopilotErrorMessages.MODEL_RESPONSE_ERROR,
                {
                    "reason": "mismatching number of embeddings",
                    "expedted_num_embeddings": len(embeddings),
                    "actual_num_embeddings": len(texts),
                },
            )
        return embeddings

    def _extract_embedding_from_api_response(self, response_data):
        embeddings = [[]] * len(response_data)
        expected_indices = set(range(len(response_data)))
        for embedding_data in response_data:
            if embedding_data.get("object", "") != "embedding":
                continue
            if not all(key in embedding_data for key in ["index", "embedding"]):
                raise RuntimeError(
                    CopilotErrorMessages.MODEL_RESPONSE_ERROR,
                    {
                        "reason": "unexpected structure of embedding entry found",
                        "entry": embedding_data,
                        "expected_keys": ["index", "embedding"],
                    },
                )
            index = embedding_data["index"]
            embedding = embedding_data["embedding"]
            if index not in expected_indices:
                raise RuntimeError(
                    CopilotErrorMessages.MODEL_RESPONSE_ERROR,
                    {
                        "reason": "embedding index not expected",
                        "index": index,
                        "expected_max_index": len(response_data) - 1,
                    },
                )
            expected_indices.remove(index)
            embeddings[index] = embedding
        if len(expected_indices) > 0:
            raise RuntimeError(
                CopilotErrorMessages.MODEL_RESPONSE_ERROR,
                {
                    "reason": "no embedding found for some indices",
                    "indices": expected_indices,
                },
            )
        return embeddings
