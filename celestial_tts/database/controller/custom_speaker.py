from uuid import UUID

from sqlmodel import select

from celestial_tts.database import Database
from celestial_tts.database.model.custom_speaker import QwenCustomSpeaker


async def select_qwen_custom_speakers(database: Database):
    async with database.async_session() as session:
        speakers = (await session.exec(select(QwenCustomSpeaker))).all()
        return speakers


async def select_qwen_custom_speaker_by_id(database: Database, id: UUID):
    async with database.async_session() as session:
        speaker = (
            await session.exec(
                select(QwenCustomSpeaker).where(QwenCustomSpeaker.id == id).limit(1)
            )
        ).first()
        return speaker


async def create_qwen_custom_speaker(database: Database, speaker: QwenCustomSpeaker):
    async with database.async_session() as session:
        session.add(speaker)
        await session.commit()
        await session.refresh(speaker)
        return speaker
