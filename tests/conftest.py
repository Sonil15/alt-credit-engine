import os

# Force the test suite to use the local SQLite database
os.environ["USE_SQLITE"] = "true"
