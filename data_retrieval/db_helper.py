import json

import ijson

from data_retrieval.labeling.guitar_identifier import GuitarIdentifier

gi = GuitarIdentifier()
unprocessed_db = "data_retrieval/reverb/reverb_raw.db"
db = "data_retrieval/reverb/reverb.db"


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
                label_quality TEXT);"""]

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
    sql = ''' INSERT OR IGNORE INTO guitars(reverb_id, reverb_make, reverb_model, reverb_finish, reverb_year, reverb_title, reverb_descr, reverb_condition, reverb_category, reverb_price, make, series, model, label_quality)
              VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?, ?) '''
    cur = conn.cursor()
    cur.execute(sql, guitar)
    conn.commit()
    return cur.lastrowid


def build_db_entry(record):
    make = gi.identify_make(record['reverb_make']) if gi.identify_make(record['reverb_make']) \
        else gi.identify_make(record['reverb_title'])

    model = gi.identify_model(make, record['reverb_model']) if gi.identify_model(make, record['reverb_model']) \
        else gi.identify_model(make, record["reverb_title"])

    series = gi.identify_series(make, record['reverb_model']) if gi.identify_series(make, record['reverb_model']) \
        else gi.identify_series(make, record["reverb_title"])

    if not series and model:
        series = gi.identify_series(make, model)

    print(f"reverb title: {record['reverb_title']}")
    print(f"rule-based inference | make: {make}, series: {series}, model: {model}")

    if not series or not model:
        label = gi.get_ollama_label(make, record["reverb_title"])
        print(f"llm-based inference | {label}")

        if not series:
            series = None if not label else label['series']
        if not model:
            model = None if not label else label['model']

    label_quality = "unassessed"

    guitar = (record['reverb_id'],
              record['reverb_make'],
              record['reverb_model'],
              record['reverb_finish'],
              record['reverb_year'],
              record['reverb_title'],
              record['reverb_descr'],
              record['reverb_condition'],
              record['reverb_category'],
              record['reverb_price'],
              make,
              series,
              model,
              label_quality)

    return guitar


def remove_processed_items():
    """
    Removes all rows from the staging database whose reverb_ids are already present in the processed database
    """
    with sqlite3.connect(unprocessed_db) as conn_unprocessed:
        deletion_cursor = conn_unprocessed.cursor()

        with sqlite3.connect(db) as conn:
            conn.row_factory = sqlite3.Row
            reading_cursor = conn.cursor()
            reading_cursor.execute("SELECT * FROM guitars")

            for row in reading_cursor:
                deletion_cursor.execute("DELETE FROM guitars WHERE reverb_id = ?", (row["reverb_id"],))

        conn_unprocessed.commit()


def process_items(query=""):
    ids = set()

    with sqlite3.connect(unprocessed_db) as conn_unprocessed:
        conn_unprocessed.row_factory = sqlite3.Row
        reading_cursor = conn_unprocessed.cursor()
        reading_cursor.execute("SELECT * FROM guitars WHERE reverb_make = ?", ("Ibanez",))
        with sqlite3.connect(db) as conn:
            for row in reading_cursor:
                # save computation time in case of duplicates in the json
                if row["reverb_id"] in ids:
                    continue
                else:
                    ids.add(row["reverb_id"])

                guitar = build_db_entry(row)
                add_guitar(conn, guitar)
                conn_unprocessed.commit()

        for id in ids:
            deletion_cursor = conn_unprocessed.cursor()
            deletion_cursor.execute("DELETE FROM guitars WHERE reverb_id = ?", (id,))
            conn_unprocessed.commit()


def postprocess_items():
    with sqlite3.connect(db) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("UPDATE guitars SET series=? WHERE series=?", ("unknown", "UNKNOWN",))
        cursor.execute("UPDATE guitars SET series=? WHERE series=?", ("unknown", "Unknown",))

        makes = [r["make"] for r in cursor.execute("SELECT DISTINCT make FROM guitars").fetchall()]

        for make in makes:
            series = cursor.execute("SELECT DISTINCT series FROM guitars WHERE make = ?", (make,))
            series = [r["series"] for r in series.fetchall()]
            print(make, ": ", series)
            print(len(series), len(set([s.lower() for s in series])))

            with open(f"data_retrieval/brand_data/{make}_data.json") as fp:
                brand_data = json.load(fp)

            # find series that deviate from master list series
            for s in series:
                if s not in brand_data["series"]:
                    if s.lower() == "unknown":
                        continue

                    # wrong lower / upper case
                    corrected = correct_case(cursor, s, brand_data)
                    if corrected: continue

                    # find closest match based on master data
                    corrected = correct_closest_match(cursor, s, brand_data)
                    if corrected: continue

                    # no successful (mass) correction possible. try individual correction
                    # or wait for master data update and try again
                    cursor.execute("UPDATE guitars SET label_quality=? WHERE series=?", ("suspect", s,))


def correct_case(cursor, current_series, brand_data):
    if current_series.lower() in [el.lower() for el in brand_data["series"]]:
        corrected_series = [el for el in brand_data["series"] if current_series.lower() == el.lower()]
        if len(corrected_series) > 1:
            print(f"Case problem in master list of {brand_data['make']}: {corrected_series}")
        else:
            corrected_series = corrected_series[0]
            cursor.execute("UPDATE guitars SET series=? WHERE series=?", (corrected_series, current_series,))

        return True

    return False


def correct_closest_match(cursor, current_series, brand_data):
    corrected_series = gi.identify_series(brand_data["make"], current_series)

    if corrected_series:
        cursor.execute("UPDATE guitars SET series=? WHERE series=?", (corrected_series, current_series,))
        return True

    return False


def init_db():
    create_sqlite_database(db)
    create_tables(db)


if __name__ == '__main__':
    # init_db()
    process_items()
    # postprocess_items()
    # remove_processed_items()
    # gi = GuitarIdentifier()
    # print(gi.identify_model("Ibanez", "Ibanez IBANEZ AZES40-TUN E-Gitarre, tungsten"))
    # label = gi.get_ollama_label("Fender", "Fender American Professional II Telecaster with Maple Fretboard Roasted Pine")
    # print("h4: ", label)
