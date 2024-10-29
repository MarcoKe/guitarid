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
    # if make == "Ibanez":
    label = gi.get_ollama_label(make, record["reverb_title"])
    print(record["reverb_title"], "\n", label)
    series = None if not label else label['series']
    model = None if not label else label['model']
    # else:
    #     series = gi.identify_series(make, record['title'])
    #     model = gi.identify_model(make, record['model']) if gi.identify_model(make, record['model']) \
    #         else gi.identify_model(make, record['title'])


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
        reading_cursor.execute("SELECT * FROM guitars WHERE reverb_make = ?", ("Kiesel", ))
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


def init_db():
    create_sqlite_database(db)
    create_tables(db)


if __name__ == '__main__':
    # init_db()
    process_items()
    # remove_processed_items()
    # gi = GuitarIdentifier()
    # label = gi.get_ollama_label("Fender", "Fender American Professional II Telecaster with Maple Fretboard Roasted Pine")
    # print("h4: ", label)