import json
import sqlite3
from db_helper import create_sqlite_database, create_tables

db = "data_retrieval/reverb/reverb_raw.db"


def add_guitar(conn, guitar):
    sql = ''' INSERT OR IGNORE INTO guitars(reverb_id, reverb_make, reverb_model, reverb_finish, reverb_year, reverb_title, reverb_descr, reverb_condition, reverb_category, reverb_price)
              VALUES(?,?,?,?,?,?,?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, guitar)
    conn.commit()
    return cur.lastrowid


def process_item(record: json):
    with sqlite3.connect(db) as conn:
            guitar = (record['id'],
                      record['make'],
                      record['model'],
                      record['finish'],
                      record['year'],
                      record['title'],
                      record['description'],
                      record['condition'],
                      record['category'],
                      record['price']['amount'] + ' ' + record['price']['currency'])
            add_guitar(conn, guitar)

def init_db():
    create_sqlite_database(db)
    create_tables(db)

if __name__ == '__main__':
    init_db()

