# Test Preset Model Generation
# Run with: hurl --test test_generate.hurl
# Note: Requires the server to be running on 127.0.0.1:8080

# 1. Basic generation with preset model and English language
POST http://127.0.0.1:8080/generate
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "text": "Hello, this is a test of the text to speech system.",
  "language": "en",
  "speaker": "Vivian"
}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.wavs" isCollection
jsonpath "$.wavs[0]" exists
jsonpath "$.sampling_rate" exists
jsonpath "$.sampling_rate" == 12000


# 2. Generation with multiple text inputs (should return multiple wavs)
POST http://127.0.0.1:8080/generate
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "text": ["First sentence.", "Second sentence.", "Third sentence."],
  "language": "en",
  "speaker": "Eric"
}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.wavs" isCollection
jsonpath "$.wavs" count == 3
jsonpath "$.sampling_rate" == 12000


# 3. Generation with instruct parameter
POST http://127.0.0.1:8080/generate
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "text": "This voice should sound happy and enthusiastic!",
  "language": "en",
  "speaker": "Serena",
  "instruct": "Happy and enthusiastic"
}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.wavs[0]" exists
jsonpath "$.sampling_rate" == 12000


# 4. Generation with all optional parameters
POST http://127.0.0.1:8080/generate
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "text": "Testing with all parameters set.",
  "language": "en",
  "speaker": "Dylan",
  "instruct": "Clear and professional",
  "top_k": 50,
  "top_p": 0.9,
  "temperature": 0.8,
  "repetition_penalty": 1.1,
  "max_new_tokens": 2048
}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.wavs[0]" exists
jsonpath "$.sampling_rate" == 12000


# 5. Test different speakers - Ryan
POST http://127.0.0.1:8080/generate
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "text": "Testing Ryan's voice.",
  "language": "en",
  "speaker": "Ryan"
}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.wavs[0]" exists


# 6. Test different speakers - Aiden
POST http://127.0.0.1:8080/generate
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "text": "Testing Aiden's voice.",
  "language": "en",
  "speaker": "Aiden"
}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.wavs[0]" exists


# 7. Test different speakers - Uncle_Fu
POST http://127.0.0.1:8080/generate
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "text": "Testing Uncle Fu's voice.",
  "language": "en",
  "speaker": "Uncle_Fu"
}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.wavs[0]" exists


# 8. Test Chinese language (zh)
POST http://127.0.0.1:8080/generate
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "text": "你好，这是中文测试。",
  "language": "zh",
  "speaker": "Vivian"
}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.wavs[0]" exists
jsonpath "$.sampling_rate" == 12000


# 9. Test Japanese language (ja)
POST http://127.0.0.1:8080/generate
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "text": "こんにちは、これは日本語のテストです。",
  "language": "ja",
  "speaker": "Ono_Anna"
}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.wavs[0]" exists
jsonpath "$.sampling_rate" == 12000


# 10. Test Korean language (ko)
POST http://127.0.0.1:8080/generate
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "text": "안녕하세요, 이것은 한국어 테스트입니다.",
  "language": "ko",
  "speaker": "Sohee"
}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.wavs[0]" exists
jsonpath "$.sampling_rate" == 12000


# 11. Test German language (de)
POST http://127.0.0.1:8080/generate
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "text": "Hallo, das ist ein deutscher Test.",
  "language": "de",
  "speaker": "Eric"
}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.wavs[0]" exists
jsonpath "$.sampling_rate" == 12000


# 12. Test French language (fr)
POST http://127.0.0.1:8080/generate
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "text": "Bonjour, ceci est un test en français.",
  "language": "fr",
  "speaker": "Serena"
}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.wavs[0]" exists
jsonpath "$.sampling_rate" == 12000


# 13. Test Russian language (ru)
POST http://127.0.0.1:8080/generate
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "text": "Привет, это русский тест.",
  "language": "ru",
  "speaker": "Dylan"
}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.wavs[0]" exists
jsonpath "$.sampling_rate" == 12000


# 14. Test Portuguese language (pt)
POST http://127.0.0.1:8080/generate
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "text": "Olá, este é um teste em português.",
  "language": "pt",
  "speaker": "Ryan"
}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.wavs[0]" exists
jsonpath "$.sampling_rate" == 12000


# 15. Test Spanish language (es)
POST http://127.0.0.1:8080/generate
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "text": "Hola, esta es una prueba en español.",
  "language": "es",
  "speaker": "Aiden"
}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.wavs[0]" exists
jsonpath "$.sampling_rate" == 12000


# 16. Test Italian language (it)
POST http://127.0.0.1:8080/generate
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "text": "Ciao, questo è un test in italiano.",
  "language": "it",
  "speaker": "Uncle_Fu"
}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.wavs[0]" exists
jsonpath "$.sampling_rate" == 12000


# 17. Test with unsupported language (should fail)
POST http://127.0.0.1:8080/generate
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "text": "This should fail",
  "language": "invalid_language",
  "speaker": "Vivian"
}
HTTP 400
[Asserts]
jsonpath "$.detail" contains "Unsupported language"
jsonpath "$.detail" contains "invalid_language"


# 18. Test with unsupported speaker (should fail)
POST http://127.0.0.1:8080/generate
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "text": "This should fail",
  "language": "en",
  "speaker": "NonExistentSpeaker"
}
HTTP 400
[Asserts]
jsonpath "$.detail" contains "Unsupported speaker"
jsonpath "$.detail" contains "NonExistentSpeaker"


# 19. Test with invalid model_id (should fail)
POST http://127.0.0.1:8080/generate
Content-Type: application/json
{
  "model_id": "invalid-model",
  "text": "This should fail",
  "language": "en",
  "speaker": "Vivian"
}
HTTP 500


# 20. Test with empty text (should fail due to NonEmptyStr validation)
POST http://127.0.0.1:8080/generate
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "text": "",
  "language": "en",
  "speaker": "Vivian"
}
HTTP 422


# 21. Test with remote provider (should fail - not implemented)
POST http://127.0.0.1:8080/generate
Content-Type: application/json
{
  "model_id": "some-remote-model",
  "text": "This should fail",
  "language": "en",
  "speaker": "Vivian",
  "provider": "remote"
}
HTTP 400
[Asserts]
jsonpath "$.detail" == "Remote models not implemented yet."


# 22. Test generation with custom model and UUID speaker
# Note: This assumes a custom speaker exists; in a real test you'd create one first
# This test demonstrates the generate endpoint works with custom model too
POST http://127.0.0.1:8080/generate
Content-Type: application/json
{
  "model_id": "qwen3-tts-custom",
  "text": "Testing custom model with preset speaker name.",
  "language": "en",
  "speaker": "test_speaker"
}
HTTP 400
[Asserts]
jsonpath "$.detail" contains "Unsupported speaker"


# 23. Verify base64 encoding of wavs
POST http://127.0.0.1:8080/generate
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "text": "Short test.",
  "language": "en",
  "speaker": "Vivian"
}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.wavs[0]" matches /^[A-Za-z0-9+/]+=*$/
jsonpath "$.sampling_rate" isInteger


# 24. Test with mixed language content (model should handle based on language param)
POST http://127.0.0.1:8080/generate
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "text": "Testing with mixed content: 你好 and hello",
  "language": "en",
  "speaker": "Dylan"
}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.wavs[0]" exists
jsonpath "$.sampling_rate" == 12000


# 25. Test long text generation
POST http://127.0.0.1:8080/generate
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "text": "This is a longer text to test the generation capabilities of the text-to-speech system. It should handle multiple sentences and longer paragraphs without any issues. The system should produce high-quality audio output that maintains consistency throughout the entire generation process.",
  "language": "en",
  "speaker": "Serena"
}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.wavs[0]" exists
jsonpath "$.sampling_rate" == 12000
