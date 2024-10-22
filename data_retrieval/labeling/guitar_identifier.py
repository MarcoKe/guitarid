import json


class GuitarIdentifier():
    def __init__(self, data_path = "data_retrieval/brand_data"):
        # read general brand data
        with open(f"{data_path}/brands.json") as f:
            brand_data = json.load(f)
        self.brands, self.brand_synonyms = brand_data["brands"], brand_data["synonyms"]
        self.brands.sort(key=lambda x: -len(x))

        # read brand specific data
        self.series = {}
        self.series_synonyms = {}
        self.models = {}
        unique_brands = set([b.lower() if not b in self.brand_synonyms
                             else self.brand_synonyms[b].lower() for b in self.brands])

        for brand in unique_brands:
            try:
                with open(f"{data_path}/{brand}_data.json") as f:
                    d = json.load(f)
                    self.series[brand] = d["series"]
                    self.series_synonyms[brand] = d["series_synonyms"]
                    self.models[brand] = d["models"]

            except FileNotFoundError:
                print("No data for brand: ", brand)

    def identify_series(self, make: str, model:str) -> str or None:
        """
        identifies a series id based on a predefined list of series
        :param make: series of which make
        :param model: model number
        :return: series id
        """
        if not make or not make in self.series: return None

        for s in self.series[make]:
            if s.lower() in model.lower():
                if s in self.synonyms[make]:
                    return self.synonyms[make][s]
                return s

        return None

    def identify_model(self, string: str, make: str) -> str or None:
        if not make or not make in self.models: return None

        for m in self.models[make]:
            if m.lower() in string.lower():
                return m
        return None

    def identify_make(self, string: str) -> str or None:
        for b in self.brands:
            if b.lower() in string.lower():
                return b

        return None