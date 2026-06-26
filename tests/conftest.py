import os

# Force the test suite to use the local SQLite database. This MUST happen before
# importing anything from `core`, because core.config reads USE_SQLITE at import
# time and core.database builds (and caches) the engine from it on import.
os.environ["USE_SQLITE"] = "true"

import asyncio
import pytest
from core.database import init_db

@pytest.fixture(scope="session", autouse=True)
def initialize_test_database():
    asyncio.run(init_db())
