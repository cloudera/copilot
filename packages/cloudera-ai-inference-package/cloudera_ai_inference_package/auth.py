import logging
import json
import os

def getAccessToken(jwt_path: str):
    if not os.path.exists(jwt_path):
        logging.error(f"{jwt_path} not found.")
        return None
    with open(jwt_path, 'r') as f:
        jwt_json = json.load(f)
        if "access_token" in jwt_json and jwt_json["access_token"]:
            return jwt_json["access_token"]
    logging.error(f"No access token found in {jwt_path}")
    return None
