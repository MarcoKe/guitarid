import json
import requests
import ollama
from bs4 import BeautifulSoup

ollama_model = "gemma2"
brands_path = "data_retrieval/brand_data/brands.json"


class ObjectFactory:
    def __init__(self):
        self._builders = {}

    def register_builder(self, key, builder):
        self._builders[key] = builder

    def create(self, key, **kwargs):
        builder = self._builders.get(key)
        if not builder:
            raise ValueError(key)
        return builder(**kwargs)

    def get(self, key, **kwargs):
        return self.create(key, **kwargs)


class OllamaIdentifier:
    def __init__(self, make_data: dict):
        self.make_data = make_data

    def parse_json(self, string: str) -> json:
        # LLM tends to return one of several variations, sometimes pure json, sometimes quotation marks in the beginning
        # In the following, everything before and after the actual json is discarded
        # string = string.replace("'", "\"")
        json_resp = string
        json_resp = "{" + "{".join(json_resp.split("{")[1:])
        json_resp = "}".join(json_resp.split("}")[:-1]) + "}"

        try:
            return json.loads(json_resp)
        except:
            return None

    def get_llm_response(self, text: str) -> json:
        prompt = self.make_data["labeling_prompt"]
        prompt = prompt.replace("{series}", " \n ".join(self.make_data["series"]))
        prompt = prompt.replace("{models}", " \n ".join(self.make_data["models"]))
        prompt += text

        response = ollama.chat(model=ollama_model, messages=[
            {
                "role": "user",
                "content": prompt,
                "format": "json",
                "stream": False,
                "options": {"temperature": 1.0}
            }
        ])

        return self.parse_json(response['message']['content'])

    def get_label(self, text: str) -> json or None:
        """
        get label from LLM and perform sanity check
        :param text: text from which to get the label
        :return: label
        """
        label = self.get_llm_response(text)

        return label

    def update_brand_data(self, series: str = None, model: str = None):
        if series:
            self.make_data["series"].append(series)
        if model:
            self.make_data["models"].append(model)

        out_path = "/".join(brands_path.split("/")[:-1]) + "/" + self.make_data["make"].lower() + "_data.json"
        with open(out_path, "w") as fp:
            json.dump(self.make_data, fp)


class IbanezOllamaIdentifier(OllamaIdentifier):
    def exists_in_wiki(self, guitar_model: str) -> bool:
        """
        checks whether the identified model exists in the ibanez fandom wiki. if the returned html contains the div class
        `noarticletext`, then no article is present for the given model and its existence hence unlikely
        :param guitar_model: identified guitar model
        :return: whether it exists in the fandom wiki
        """
        url = f"https://ibanez.fandom.com/wiki/{guitar_model}"
        r = requests.get(url)
        return "noarticletext" not in r.text


    def refine_series(self, guitar_model: str):
        url = f"https://ibanez.fandom.com/wiki/{guitar_model}"
        r = requests.get(url)

        soup = BeautifulSoup(r.text, "lxml")
        descr = soup.find("meta", property="og:description")

        prompt = self.make_data["series_confirmation_prompt"]
        prompt = prompt.replace("{series}", " \n ".join(self.make_data["series"]))
        prompt = prompt.replace("{models}", " \n ".join(self.make_data["models"]))
        prompt += descr["content"]

        response = ollama.chat(model=ollama_model, messages=[
            {
                "role": "user",
                "content": prompt,
                "format": "json",
                "stream": False,
                "options": {"temperature": 1.0}
            }
        ])

        return self.parse_json(response['message']['content'])['series']

    def get_label(self, text: str) -> json or None:
        """
        get label from LLM and perform sanity check
        :param text: text from which to get the label
        :return: label
        """
        label = self.get_llm_response(text)
        label['model'] = label['model'].upper()

        # sanity check
        if not self.exists_in_wiki(label['model']):
            label = self.get_llm_response(text)
            label['model'] = label['model'].upper()

            if not self.exists_in_wiki(label['model']):
                return None # give up after trying twice

        self.update_brand_data(model=label["model"])

        # if the identified series does not exist in the wiki, try extracting it from model page of the wiki
        if not self.exists_in_wiki(label['series']):
            label['series'] = self.refine_series(label['model'])
            if not self.exists_in_wiki(label['series']):
                label['series'] = None



        return label


identifiers = ObjectFactory()

with open(brands_path, "r") as f:
    brands = json.load(f)

for brand in brands["brands"]: # set default identifier for all brands
    identifiers.register_builder(brand, OllamaIdentifier)

# overwrite individualised identifiers:
identifiers.register_builder("Ibanez", IbanezOllamaIdentifier)