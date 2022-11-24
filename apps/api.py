# -*- encoding: utf-8 -*-

# database.py

import os
from dotenv import load_dotenv
import requests

load_dotenv()

HOST = os.getenv("API_HOST")
USERNAME = os.getenv("API_USERNAME")
PASSWORD = os.getenv("API_PASSWORD")
print(HOST)
print(USERNAME)
res = requests.post(HOST+'/api/login',json={"username": USERNAME, "email": PASSWORD})
token = res.text.replace('"','')
headers = {"Authorization":"Bearer "+token}
api = {"headers": headers, "host":HOST}
def get_api():
    yield api
