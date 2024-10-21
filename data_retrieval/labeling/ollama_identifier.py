import json
import requests
import ollama
from bs4 import BeautifulSoup

system_prompt_path = "data_retrieval/labeling/ollama_system_prompt_ibanez.txt"
valid_series_path = "data_retrieval/labeling/valid_series_ibanez.txt"
ollama_model = "llama3.1"
ollama_client = ollama.Client(host='http://localhost:11434')


with open(system_prompt_path, "r") as f:
    system_prompt = f.read()

with open(valid_series_path, "r") as f:
    valid_series = f.read()


def parse_json(string: str) -> json:
    string = string.replace("'", "\"")
    json_resp = string
    json_resp = "{" + "{".join(json_resp.split("{")[1:])
    json_resp = "}".join(json_resp.split("}")[:-1]) + "}"

    return json.loads(json_resp)


def get_llm_response(text: str) -> json:
    response = ollama.chat(model=ollama_model, messages=[
        {
            "role": "user",
            "content": system_prompt + text,
            "format": "json",
            "stream": False,
            "options": {"temperature": 1.0}
        }
    ])

    # LLM tends to return one of several variations, sometimes pure json, sometimes quotation marks in the beginning
    # In the following, everything before and after the actual json is discarded
    return parse_json(response['message']['content'])


def exists_in_wiki(guitar_model: str) -> bool:
    """
    checks whether the identified model exists in the ibanez fandom wiki. if the returned html contains the div class
    `noarticletext`, then no article is present for the given model and its existence hence unlikely
    :param guitar_model: identified guitar model
    :return: whether it exists in the fandom wiki
    """
    url = f"https://ibanez.fandom.com/wiki/{guitar_model}"
    r = requests.get(url)
    return "noarticletext" not in r.text


def refine_series(guitar_model: str):
    url = f"https://ibanez.fandom.com/wiki/{guitar_model}"
    r = requests.get(url)

    soup = BeautifulSoup(r.text, "lxml")
    descr = soup.find("meta", property="og:description")

    response = ollama.chat(model=ollama_model, messages=[
        {
            "role": "user",
            "content": "What is the series of the guitar model according to the following text? Valid Series are: " +
                       valid_series + " \n Here is the text: " + descr["content"] + "\n Respond in json from according to"
                       "the following schema: {'series': 'the series you found out'}. Give only the json as a response, nothing else.",
            "format": "json",
            "stream": False,
            "options": {"temperature": 1.0}
        }
    ])

    return parse_json(response['message']['content'])['series']


def get_label(text: str) -> json or None:
    """
    get label from LLM and perform sanity check
    :param text: text from which to get the label
    :return: label
    """
    label = get_llm_response(text)

    # sanity check
    if not exists_in_wiki(label['model']):
        label = get_llm_response(text)
        if not exists_in_wiki(label['model']):
            return None # give up after trying twice

    # if the identified series does not exist in the wiki, try extracting it from model page of the wiki
    if not exists_in_wiki(label['series']):
        label['series'] = refine_series(label['model'])
        if not exists_in_wiki(label['series']):
            label['series'] = None

    return label

#
# prompt = 'Ibanez SEW761FM-NTF Standard 2024 - Natural Flat '
# label = get_label(prompt)
# print(label)


# prompt = 'Ibanez AF95-DA Artcore Express. 6-Str Dark Amber 2024'
# label = get_llm_label(prompt)
# print(label)
# print(exists_in_wiki(label['model']))
#
# prompt = 'Ibanez RG370DX with a nice hat'
# label = get_llm_label(prompt)
# print(label)
# print(exists_in_wiki(label['model']))
#
# prompt = 'Ibanez RG1451 in white with a turqoise pickguard'
# label = get_llm_label(prompt)
# print(label)
# print(exists_in_wiki(label['model']))