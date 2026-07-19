import sqlite3
import json

try:
    conn = sqlite3.connect('altcredit.db')
    c = conn.cursor()
    c.execute("SELECT user_id, cohort, requested_amount FROM application_intake")
    rows = c.fetchall()
    print("ApplicationIntakes in DB:")
    for r in rows:
        print(r)
except Exception as e:
    print(e)
