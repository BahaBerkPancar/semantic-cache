from alembic import context
from sqlalchemy import create_engine, pool

from semantic_cache import models  # noqa: F401  (registers the tables)
from semantic_cache.config import settings
from semantic_cache.db import Base

config = context.config
url = config.get_main_option("sqlalchemy.url") or settings.database_url


def run_offline() -> None:
    context.configure(url=url, target_metadata=Base.metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_online() -> None:
    engine = create_engine(url, poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=Base.metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_offline()
else:
    run_online()
