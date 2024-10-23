import pathlib
import shutil
import requests
import raw_db_helper


headers = {"Content-Type": "application/hal+json",
            "Accept": "application/hal+json",
            "Accept-Version": "3.0"}

json_path = "data_retrieval/reverb/json/"
img_path = "data_retrieval/reverb/img/all"


def retrieve_data():
    r = requests.get(f"https://api.reverb.com/api/listings?query=gibson&product_type=electric-guitars", headers=headers)

    data = r.json()
    links = data["_links"]


    i = 0
    while "next" in links:
        try:
            print("scraping page ", i)
            r = requests.get(links["next"]["href"], headers=headers)
            data_ = r.json()
            links = data_["_links"]
            data["listings"].extend(data_["listings"])
            i += 1

            # extract relevant data from overall response and save image
            extract_training_data(data)
        except:
            print("Problem on page ", i)
            continue


def extract_training_data(data):
    for listing in data["listings"]:
        item_data = {
            "id": listing["id"],
            "make": listing["make"],
            "model": listing["model"],
            "finish": listing["finish"],
            "year": listing["year"],
            "title": listing["title"],
            "description": listing["description"],
            "condition": listing["condition"]["display_name"],
            "category": listing["categories"][0]["full_name"].replace(" ", ""),
            "price": {
                "amount": listing["price"]["amount"],
                "currency": listing["price"]["currency"],
            }}

        raw_db_helper.process_item(item_data)
        img_url = listing["photos"][0]["_links"]["full"]["href"]

        # save in separate category directories (electric-guitar/solid-body, parts/knobs, etc)
        category_path = img_path
        p = pathlib.Path(category_path)
        p.mkdir(parents=True, exist_ok=True)

        response = requests.get(img_url, headers=headers, stream=True)
        with open(f"{category_path}/{listing['id']}.jpg", 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)

    return True


retrieve_data()