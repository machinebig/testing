from alembic import context
from sqlalchemy import engine_from_config, pool
from core.config import settings
from models.user import User
from models.project import Project
from models.validation import Validation

config = context.config
config.set_main_option('sqlalchemy.url', settings.DATABASE_URL)
connectable = engine_from_config(config.get_section(config.config_group), prefix='sqlalchemy.', poolclass=pool.NullPool)
with connectable.connect() as connection:
    context.configure(connection=connection, target_metadata=Base.metadata)
    with context.begin_transaction():
        context.run_migrations()
