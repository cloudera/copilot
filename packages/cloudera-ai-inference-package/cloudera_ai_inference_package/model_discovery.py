import json
import os

from cloudera_ai_inference_package.error_messages import CopilotErrorMessages


def getCopilotModels(config_dir, model_type):
    if model_type not in ["inference", "embedding"]:
        raise ValueError(
            CopilotErrorMessages.INTERNAL_ERROR,
            {
                "reason": "unknown  model type when loading copilot models",
                "type": model_type,
            },
        )
    if not config_dir or not os.path.exists(config_dir):
        return [], []

    ai_inference_models = []
    with open(config_dir) as f:
        copilot_config = json.load(f)
        if (
            copilot_config
            and "aiInferenceModels" in copilot_config
            and copilot_config["aiInferenceModels"]
        ):
            ai_inference_models = copilot_config["aiInferenceModels"]

    embeddingModelFilter = lambda model: model["endpoint"].endswith("/embeddings")
    inferenceModelFilter = lambda model: not embeddingModelFilter(model)
    if model_type == "inference":
        ai_inference_models = list(filter(inferenceModelFilter, ai_inference_models))
    elif model_type == "embedding":
        ai_inference_models = list(filter(embeddingModelFilter, ai_inference_models))

    models = []
    for ai_inference_model in ai_inference_models:
        models.append(ai_inference_model["name"])

    return ai_inference_models, models
