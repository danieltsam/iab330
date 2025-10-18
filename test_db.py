from db import get_db
try:
    get_db()
    print("MongoDB connection OK")
except Exception as e:
    print("MongoDB connection FAILED:", e)
