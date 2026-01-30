"""
Test the OpenAI-compatible /v1/audio/speech endpoint using the official OpenAI client.

Environment variables:
    CELESTIAL_TTS_TOKEN: Authentication token for the API
    CELESTIAL_TTS_BASE_URL: Base URL for the API (default: http://127.0.0.1:8080/api/v1)
"""

import os
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import pytest
import soundfile as sf
from openai import OpenAI

# Configuration
BASE_URL = os.getenv("CELESTIAL_TTS_BASE_URL", "http://127.0.0.1:8080/api/v1")
TOKEN = os.getenv("CELESTIAL_TTS_TOKEN")

Format = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]
OpenAiVoice = Literal["alloy", "nova", "fable", "Vivian"]

if not TOKEN:
    pytest.skip(
        "CELESTIAL_TTS_TOKEN environment variable not set",
        allow_module_level=True,
    )


@pytest.fixture
def client() -> OpenAI:
    """Create an OpenAI client configured for Celestial TTS."""
    return OpenAI(base_url=BASE_URL, api_key=TOKEN)


class TestBasicSpeech:
    """Test basic speech generation functionality."""

    def test_simple_generation(self, client: OpenAI, tmp_path: Path) -> None:
        """Test simple speech generation with default parameters."""
        output_file = tmp_path / "test_simple.mp3"

        with client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="alloy",
            input="Hello, this is a test of the OpenAI compatible API.",
        ) as response:
            response.stream_to_file(output_file)

        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_with_tts_1_hd_model(self, client: OpenAI, tmp_path: Path) -> None:
        """Test with tts-1-hd model."""
        output_file = tmp_path / "test_hd.mp3"

        with client.audio.speech.with_streaming_response.create(
            model="tts-1-hd",
            voice="nova",
            input="Testing the high definition model.",
        ) as response:
            response.stream_to_file(output_file)

        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_with_native_model_id(self, client: OpenAI, tmp_path: Path) -> None:
        """Test with native model ID instead of OpenAI model name."""
        output_file = tmp_path / "test_native.mp3"

        with client.audio.speech.with_streaming_response.create(
            model="qwen3-tts-0.6b-preset",
            voice="Vivian",
            input="Testing with native model ID and speaker.",
        ) as response:
            response.stream_to_file(output_file)

        assert output_file.exists()
        assert output_file.stat().st_size > 0


class TestVoiceMapping:
    """Test OpenAI voice name mapping to native speakers."""

    @pytest.mark.parametrize(
        "voice",
        ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
    )
    def test_openai_voices(self, client: OpenAI, voice: str, tmp_path: Path) -> None:
        """Test all OpenAI voice names."""
        output_file = tmp_path / f"test_{voice}.mp3"

        with client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice=voice,
            input=f"Testing the {voice} voice.",
        ) as response:
            response.stream_to_file(output_file)

        assert output_file.exists()
        assert output_file.stat().st_size > 0

    @pytest.mark.parametrize(
        "speaker",
        ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Sohee"],
    )
    def test_native_speakers(
        self, client: OpenAI, speaker: str, tmp_path: Path
    ) -> None:
        """Test native speaker names."""
        output_file = tmp_path / f"test_{speaker}.mp3"

        with client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice=speaker,
            input=f"Testing {speaker}'s voice.",
        ) as response:
            response.stream_to_file(output_file)

        assert output_file.exists()
        assert output_file.stat().st_size > 0


class TestAudioFormats:
    """Test different audio output formats."""

    @pytest.mark.parametrize(
        "format,extension,expected_format",
        [
            ("mp3", "mp3", "MP3"),
            ("wav", "wav", "WAV"),
            ("opus", "opus", "OGG"),  # Opus uses OGG container
            ("flac", "flac", "FLAC"),
            ("pcm", "pcm", None),  # PCM is raw data, no header to validate
        ],
    )
    def test_audio_formats(
        self,
        client: OpenAI,
        format: Format,
        extension: str,
        expected_format: Optional[str],
        tmp_path: Path,
    ) -> None:
        """Test different audio output formats."""
        output_file = tmp_path / f"test.{extension}"

        with client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="alloy",
            input=f"Testing {format.upper()} format.",
            response_format=format,
        ) as response:
            response.stream_to_file(output_file)

        assert output_file.exists()
        assert output_file.stat().st_size > 0

        # Validate actual audio format (PCM has no header to validate)
        if expected_format is not None:
            info = sf.info(output_file)
            assert info.format == expected_format, (
                f"Expected {expected_format} format, got {info.format}"
            )


class TestSpeedParameter:
    """Test speed parameter functionality."""

    @pytest.mark.parametrize(
        "speed",
        [0.25, 0.5, 1.0, 1.5, 2.0, 4.0],
    )
    def test_valid_speeds(self, client: OpenAI, speed: float, tmp_path: Path) -> None:
        """Test valid speed values."""
        with client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="alloy",
            input="Testing speech speed.",
            speed=speed,
        ) as response:
            output_file = tmp_path / f"test_speed_{speed}.mp3"
            response.stream_to_file(output_file)

        assert output_file.exists()
        assert output_file.stat().st_size > 0


class TestMultilingual:
    """Test multilingual content generation."""

    @pytest.mark.parametrize(
        "language,text",
        [
            ("English", "Hello, this is a test."),
            ("Chinese", "你好，这是中文测试。"),
            ("Japanese", "こんにちは、これは日本語のテストです。"),
            ("Korean", "안녕하세요, 이것은 한국어 테스트입니다."),
            ("Spanish", "Hola, esta es una prueba en español."),
            ("French", "Bonjour, ceci est un test en français."),
            ("German", "Hallo, das ist ein deutscher Test."),
            ("Portuguese", "Olá, este é um teste em português."),
            ("Russian", "Привет, это русский тест."),
            ("Italian", "Ciao, questo è un test in italiano."),
        ],
    )
    def test_multilingual_generation(
        self, client: OpenAI, language: str, text: str, tmp_path: Path
    ) -> None:
        """Test speech generation in different languages."""
        output_file = tmp_path / f"test_{language.lower()}.mp3"

        with client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="alloy",
            input=text,
        ) as response:
            response.stream_to_file(output_file)

        assert output_file.exists()
        assert output_file.stat().st_size > 0


class TestErrorHandling:
    """Test error handling and validation."""

    def test_empty_input(self, client: OpenAI) -> None:
        """Test that empty input raises an error."""
        with pytest.raises(Exception):
            client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input="",
            )

    def test_invalid_voice(self, client: OpenAI) -> None:
        """Test that invalid voice name raises an error."""
        with pytest.raises(Exception):
            client.audio.speech.create(
                model="tts-1",
                voice="nonexistent_voice",
                input="This should fail.",
            )

    def test_invalid_format(self, client: OpenAI) -> None:
        """Test that invalid response format raises an error."""
        with pytest.raises(Exception):
            client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input="This should fail.",
                response_format="invalid_format",  # pyright: ignore[reportArgumentType]
            )

    def test_speed_too_low(self, client: OpenAI) -> None:
        """Test that speed below minimum raises an error."""
        with pytest.raises(Exception):
            client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input="This should fail.",
                speed=0.1,
            )

    def test_speed_too_high(self, client: OpenAI) -> None:
        """Test that speed above maximum raises an error."""
        with pytest.raises(Exception):
            client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input="This should fail.",
                speed=5.0,
            )


class TestEdgeCases:
    """Test edge cases and special content."""

    def test_long_text(self, client: OpenAI, tmp_path: Path) -> None:
        """Test with longer text content."""
        long_text = (
            "This is a longer text to test the speech generation capabilities. "
            "It should handle multiple sentences and longer paragraphs without any issues. "
            "The system should produce high-quality audio output that maintains consistency "
            "throughout the entire generation process. This helps ensure that the API can "
            "handle realistic use cases where users need to generate longer form audio content."
        )
        output_file = tmp_path / "test_long.mp3"

        with client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="nova",
            input=long_text,
        ) as response:
            response.stream_to_file(output_file)

        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_special_characters(self, client: OpenAI, tmp_path: Path) -> None:
        """Test with special characters and punctuation."""
        text = (
            "Testing with special characters: @#$%^&*()!? and numbers 123456. "
            'She said, "Hello!" and I replied, "It\'s a beautiful day, isn\'t it?"'
        )
        output_file = tmp_path / "test_special.mp3"

        with client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="alloy",
            input=text,
        ) as response:
            response.stream_to_file(output_file)

        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_mixed_language(self, client: OpenAI, tmp_path: Path) -> None:
        """Test with mixed language content."""
        text = "Hello, this is English. 你好，这是中文。"
        output_file = tmp_path / "test_mixed.mp3"

        with client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="alloy",
            input=text,
        ) as response:
            response.stream_to_file(output_file)

        assert output_file.exists()
        assert output_file.stat().st_size > 0


class TestCombinedParameters:
    """Test combinations of parameters."""

    def test_all_parameters(self, client: OpenAI, tmp_path: Path) -> None:
        """Test with all optional parameters specified."""
        output_file = tmp_path / "test_all_params.wav"

        with client.audio.speech.with_streaming_response.create(
            model="tts-1-hd",
            voice="nova",
            input="Testing with all parameters specified.",
            response_format="wav",
            speed=1.25,
        ) as response:
            response.stream_to_file(output_file)

        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_different_voice_format_combinations(
        self, client: OpenAI, tmp_path: Path
    ) -> None:
        """Test different combinations of voices and formats."""

        combinations: List[Tuple[OpenAiVoice, Format]] = [
            ("alloy", "mp3"),
            ("nova", "wav"),
            ("fable", "flac"),
            ("Vivian", "opus"),
        ]

        for voice, format in combinations:
            output_file = tmp_path / f"test_{voice}_{format}.{format}"

            with client.audio.speech.with_streaming_response.create(
                model="tts-1",
                voice=voice,
                input=f"Testing {voice} with {format} format.",
                response_format=format,
            ) as response:
                response.stream_to_file(output_file)

            assert output_file.exists()
            assert output_file.stat().st_size > 0
