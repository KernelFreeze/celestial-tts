import ssl

from sqlalchemy import URL
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from .model.custom_speaker import QwenCustomSpeaker

__all__ = ["Database", "QwenCustomSpeaker"]


class Database:
    def __init__(self, url: str | URL):
        # Add connection arguments for better timeout handling and SSL
        connect_args = {}
        if isinstance(url, str) and "postgresql" in url:
            # Create SSL context for secure connection
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE

            connect_args = {
                "ssl": ssl_context,
                "timeout": 30,  # 30 second connection timeout
                "command_timeout": 10,
            }

        self.engine = create_async_engine(
            url,
            connect_args=connect_args,
            pool_pre_ping=True,  # Verify connections before using them
        )
        self.async_session = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def init_db(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(SQLModel.metadata.create_all)

    async def close(self):
        await self.engine.dispose()
