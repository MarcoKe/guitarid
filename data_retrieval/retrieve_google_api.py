import pathlib
import requests
import json

import urllib.request

with open("credentials.json") as f:
    credentials = json.load(f)

def find_images(query):
    # replace with your own CSE ID and API key
    cse_id = credentials['cse_id']
    api_key = credentials['api_key']

    # 1, 11, 21, 31, 41, 51, 61, 71, 81, 91
    for i in range(10):
        start = 1 + (i*10)
        url = f"https://www.googleapis.com/customsearch/v1?q={query}&imgSize=huge&searchType=image&key={api_key}&cx={cse_id}&start={start}&fileType=jpg"

        response = requests.get(url)
        response.raise_for_status()
        if i > 0:
              search_results["items"].extend(response.json()["items"])
        else:
            search_results = response.json()

    p = pathlib.Path(query + "/")
    p.mkdir(parents=True, exist_ok=True)

    with open(query + '/data.json', 'w+') as f:
        json.dump(search_results, f)


from collections.abc import MutableMapping

def flatten(dictionary, parent_key='', separator='_'):
    # from: https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)

def save_images(query):
    with open(query + "/data.json") as f:
        data = json.load(f)

    for i, item in enumerate(data["items"]):
        save_image = True

        # only save image if each term of the search query occurs at least somewhere in some field of the result dict
        # for substring in query.split(" "):
        #     present = False
        #     item = flatten(item)
        #     for key in item.keys():
        #         if substring in str(item[key]).replace(" ", ""):
        #             present = True
        #             break
        #
        #     if not present:
        #         save_image = False

        if save_image:
            filetype = item['link'].split('/')[-1].split('.')[-1]
            try:
                urllib.request.urlretrieve(item["link"], f"{query}/{i}.jpg")
            except:
                print(f"couldnt get {item['link']}")


query = "Fender Player Telecaster"
find_images(query)
save_images(query)
