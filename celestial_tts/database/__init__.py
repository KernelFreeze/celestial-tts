from sqlalchemy import URL
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession


class Database:
    def __init__(self, url: str | URL):
        self.engine = create_async_engine("sqlite+aiosqlite:///database.db")
        self.async_session = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def init_db(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)


def create_test_database():
    sqlite_file_name = "database.db"
    sqlite_url = f"sqlite:///{sqlite_file_name}"

    return Database(sqlite_url)
