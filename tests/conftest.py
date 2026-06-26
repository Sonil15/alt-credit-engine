import os
import asyncio
import pytest
from core.database import init_db

# Force the test suite to use the local SQLite database
os.environ["USE_SQLITE"] = "true"

@pytest.fixture(scope="session", autouse=True)
def initialize_test_database():
    asyncio.run(init_db())
