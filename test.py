import requests
import json


# replace with your own CSE ID and API key
cse_id = "c1d8bbf3f47a54b00"
api_key = "AIzaSyAkXXtV0Os2WoFZNlNUBNoII_jMudFG1Ik"

url = f"https://www.googleapis.com/customsearch/v1?q=Ibanez RG370DX&imgSize=huge&searchType=image&key={api_key}&cx={cse_id}"

response = requests.get(url)
response.raise_for_status()


search_results = response.json()

with open('data.json', 'w') as f:
    json.dump(search_results, f)

image_url = search_results['items'][0]['link']

print('Image URL:', image_url)