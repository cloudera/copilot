import os
from logging import Logger

class FakeCMLAPI(object):
    """
    Fake CML API client for testing purposes.  The real API call should look like this:

    import cmlapi
    api_client = cmlapi.default_client()
    api_client.send_copilot_event(req)
    """
    class client(object):
        def send_copilot_event(self, req):
            logger = Logger("copilot")
            if "application_id" in req.keys():
                logger.info(f"application_id: {req['application_id']}")
            if "engine_id" in req.keys():
                logger.info(f"engine_id: {UsageTracker.engine_id}")
            if "event_details" in req.keys():
                logger.info(f"event_details: {req['event_details']}")
            if "event_type" in req.keys():
                logger.info(f"event_type: {req['event_type']}")
            if "include_selection" in req.keys():
                logger.info(f"include_selection: {req['include_selection']}")
            if "model_name" in req.keys():
                logger.info(f"model_name: {req['model_name']}")
            if "model_provider_id" in req.keys():
                logger.info(f"model_provider_id: {req['model_provider_id']}")
            if "model_type" in req.keys():
                logger.info(f"model_type: {req['model_type']}")
            if "prompt_word_count" in req.keys():
                logger.info(f"prompt_word_count: {req['prompt_word_count']}")

    def default_client():
        return FakeCMLAPI.client()

try:
    import cmlapi
except ImportError as e:
    cmlapi = FakeCMLAPI

class Singleton:
    # Singleton pattern
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class UsageTracker(object):
    __metaclass__ = Singleton

    engine_id = os.getenv("CDSW_ENGINE_ID")

    def _SendCopilotEvent(self,  req):
        req["engine_id"] = self.engine_id
        api_client = cmlapi.default_client()
        api_client.send_copilot_event(req)