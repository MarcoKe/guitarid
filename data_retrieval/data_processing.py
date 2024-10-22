import ijson

from data_retrieval.labeling.guitar_identifier import GuitarIdentifier
from data_retrieval.labeling.ollama_identifier import get_label

gi = GuitarIdentifier()


def create_sqlite_database(filename):
    """ create a database connection to an SQLite database """
    conn = None
    try:
        conn = sqlite3.connect(filename)
        print(sqlite3.sqlite_version)
    except sqlite3.Error as e:
        print(e)
    finally:
        if conn:
            conn.close()


import sqlite3


def create_tables(db):
    sql_statements = [
        """CREATE TABLE IF NOT EXISTS guitars (
                id INTEGER PRIMARY KEY, 
                reverb_id INTEGER NOT NULL UNIQUE, 
                reverb_make TEXT, 
                reverb_model TEXT,
                reverb_finish TEXT,
                reverb_year TEXT,
                reverb_title TEXT,
                reverb_descr TEXT,
                reverb_condition TEXT,
                reverb_category TEXT,
                reverb_price TEXT,
                make TEXT, 
                series TEXT,
                model TEXT,
                img_path TEXT);"""]

    # create a database connection
    try:
        with sqlite3.connect(db) as conn:
            cursor = conn.cursor()
            for statement in sql_statements:
                cursor.execute(statement)

            conn.commit()

    except sqlite3.Error as e:
        print(e)


def add_guitar(conn, guitar):
    sql = ''' INSERT OR IGNORE INTO guitars(reverb_id, reverb_make, reverb_model, reverb_finish, reverb_year, reverb_title, reverb_descr, reverb_condition, reverb_category, reverb_price, make, series, model, img_path)
              VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?, ?) '''
    cur = conn.cursor()
    cur.execute(sql, guitar)
    conn.commit()
    return cur.lastrowid


def build_db_entry(record):
    make = gi.identify_make(record['make']) if gi.identify_make(record['make']) else gi.identify_make(record['title'])
    if make == "Ibanez":
        label = get_label(record["title"])
        print(record["title"], "\n", label)
        series = None if not label else label['series']
        model = None if not label else label['model']
    else:
        series = gi.identify_series(make, record['title'])
        model = gi.identify_model(make, record['model']) if gi.identify_model(make, record['model']) \
            else gi.identify_model(make, record['title'])


    img_path = f"{make}_{series}_{model}/{record['id']}.jpg"

    guitar = (record['id'],
              record['make'],
              record['model'],
              record['finish'],
              record['year'],
              record['title'],
              record['description'],
              record['condition'],
              record['category'],
              record['price']['amount'] + ' ' + record['price']['currency'],
              make,
              series,
              model,
              img_path)

    return guitar


def process_items(path="data_retrieval/reverb/json/_all.json"):
    db = "data_retrieval/reverb/reverb.db"
    create_sqlite_database(db)
    create_tables(db)

    ids = set()
    with sqlite3.connect(db) as conn:
        with open(path, "rb") as f:

            for record in ijson.items(f, "items.item"):
                # save computation time in case of duplicates in the json
                if record['id'] in ids:
                    continue
                else:
                    ids.add(record['id'])

                guitar = build_db_entry(record)
                add_guitar(conn, guitar)


if __name__ == '__main__':
    process_items()