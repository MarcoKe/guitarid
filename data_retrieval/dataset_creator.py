import tqdm
import pathlib
import shutil
import sqlite3
import numpy as np

dataset_path = "datasets/reverb_simple/"
raw_data_path = "data_retrieval/reverb/reverb_.db"
class_threshold = 100  # classes with fewer examples will be discarded
val_proportion = 0.3
class_fields = ["make", "series"]  # order matters: from coarse to fine
multi_task_classification = False  # if true, multiple labels (one for each field) will be generated for each example. if false, fields will be concatenated


img_path = "/".join(raw_data_path.split("/")[:-1]) + "/img/all/"

def make_path(record):
    sep = "_"
    if multi_task_classification: sep = "/"

    class_str = sep.join([record[field] for field in class_fields])
    path = pathlib.Path(dataset_path + "train/" + class_str + "/" + str(record["reverb_id"]) + ".jpg")
    path.parent.mkdir(parents=True, exist_ok=True)

    return path

# fetch data from database where class threshold is met
with sqlite3.connect(raw_data_path) as conn:
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(f"SELECT reverb_id, make, series, model FROM guitars WHERE {class_fields[-1]} != 'unknown' AND {class_fields[-1]} IN "
                   f"(SELECT {class_fields[-1]} FROM guitars GROUP BY {class_fields[-1]} HAVING COUNT ({class_fields[-1]}) >= {class_threshold})")

    # copy images into dataset location and structure
    print("Moving files to dataset location..")
    for row in tqdm.tqdm(cursor):
        path = make_path(row)
        shutil.copy(img_path + str(row["reverb_id"]) + ".jpg", path)


# create validation set
train_files = list(pathlib.Path(dataset_path).rglob("*.jpg"))
selected_files = np.random.choice(len(train_files), int(len(train_files) * val_proportion), replace=False) # ideally we would select for each class separately
print(len(selected_files), len(set(selected_files)))

print("Creating validation set..")
for f in tqdm.tqdm(selected_files):
    file = str(train_files[f])
    val_path = file.replace("train", "val")
    pathlib.Path(val_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(file), str(val_path))
