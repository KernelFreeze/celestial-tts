# Test Preset Model Generation
# Run with: hurl --variables-file tests/hurl/vars.env --test tests/hurl/test_generate.hurl
# Note: Requires the server to be running on 127.0.0.1:8080

# 1. Basic generation with preset model and english language
POST {{base_url}}/api/v1/generate
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "text": "Hello, this is a test of the text to speech system.",
  "language": "english",
  "speaker": "Vivian"
}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.wavs" isCollection
jsonpath "$.wavs[0]" exists
jsonpath "$.sampling_rate" exists
jsonpath "$.sampling_rate" == 24000


# 2. Generation with multiple text inputs (should return multiple wavs)
POST {{base_url}}/api/v1/generate
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "text": ["First sentence.", "Second sentence.", "Third sentence."],
  "language": "english",
  "speaker": "Eric"
}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.wavs" isCollection
jsonpath "$.wavs" count == 3
jsonpath "$.sampling_rate" == 24000


# 3. Generation with instruct parameter
POST {{base_url}}/api/v1/generate
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "text": "This voice should sound happy and enthusiastic!",
  "language": "english",
  "speaker": "Serena",
  "instruct": "Happy and enthusiastic"
}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.wavs[0]" exists
jsonpath "$.sampling_rate" == 24000


# 4. Generation with all optional parameters
POST {{base_url}}/api/v1/generate
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "text": "Testing with all parameters set.",
  "language": "english",
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
jsonpath "$.sampling_rate" == 24000


# 5. Test different speakers - Ryan
POST {{base_url}}/api/v1/generate
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "text": "Testing Ryan's voice.",
  "language": "english",
  "speaker": "Ryan"
}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.wavs[0]" exists


# 6. Test different speakers - Aiden
POST {{base_url}}/api/v1/generate
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "text": "Testing Aiden's voice.",
  "language": "english",
  "speaker": "Aiden"
}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.wavs[0]" exists


# 7. Test different speakers - Uncle_Fu
POST {{base_url}}/api/v1/generate
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "text": "Testing Uncle Fu's voice.",
  "language": "english",
  "speaker": "Uncle_Fu"
}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.wavs[0]" exists


# 8. Test Chinese language (zh)
POST {{base_url}}/api/v1/generate
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "text": "你好，这是中文测试。",
  "language": "chinese",
  "speaker": "Vivian"
}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.wavs[0]" exists
jsonpath "$.sampling_rate" == 24000


# 9. Test Japanese language (ja)
POST {{base_url}}/api/v1/generate
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "text": "こんにちは、これは日本語のテストです。",
  "language": "japanese",
  "speaker": "Ono_Anna"
}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.wavs[0]" exists
jsonpath "$.sampling_rate" == 24000


# 10. Test Korean language (ko)
POST {{base_url}}/api/v1/generate
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "text": "안녕하세요, 이것은 한국어 테스트입니다.",
  "language": "korean",
  "speaker": "Sohee"
}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.wavs[0]" exists
jsonpath "$.sampling_rate" == 24000


# 11. Test German language (de)
POST {{base_url}}/api/v1/generate
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "text": "Hallo, das ist ein deutscher Test.",
  "language": "german",
  "speaker": "Eric"
}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.wavs[0]" exists
jsonpath "$.sampling_rate" == 24000


# 12. Test French language (fr)
POST {{base_url}}/api/v1/generate
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "text": "Bonjour, ceci est un test en français.",
  "language": "french",
  "speaker": "Serena"
}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.wavs[0]" exists
jsonpath "$.sampling_rate" == 24000


# 13. Test Russian language (ru)
POST {{base_url}}/api/v1/generate
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "text": "Привет, это русский тест.",
  "language": "russian",
  "speaker": "Dylan"
}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.wavs[0]" exists
jsonpath "$.sampling_rate" == 24000


# 14. Test Portuguese language (pt)
POST {{base_url}}/api/v1/generate
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "text": "Olá, este é um teste em português.",
  "language": "portuguese",
  "speaker": "Ryan"
}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.wavs[0]" exists
jsonpath "$.sampling_rate" == 24000


# 15. Test Spanish language (es)
POST {{base_url}}/api/v1/generate
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "text": "Hola, esta es una prueba en español.",
  "language": "spanish",
  "speaker": "Aiden"
}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.wavs[0]" exists
jsonpath "$.sampling_rate" == 24000


# 16. Test Italian language (it)
POST {{base_url}}/api/v1/generate
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "text": "Ciao, questo è un test in italiano.",
  "language": "italian",
  "speaker": "Uncle_Fu"
}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.wavs[0]" exists
jsonpath "$.sampling_rate" == 24000


# 17. Test with unsupported language (should fail)
POST {{base_url}}/api/v1/generate
Authorization: Bearer {{token}}
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
POST {{base_url}}/api/v1/generate
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "text": "This should fail",
  "language": "english",
  "speaker": "NonExistentSpeaker"
}
HTTP 400
[Asserts]
jsonpath "$.detail" contains "Unsupported speaker"
jsonpath "$.detail" contains "NonExistentSpeaker"


# 19. Test with invalid model_id (should fail)
POST {{base_url}}/api/v1/generate
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model_id": "invalid-model",
  "text": "This should fail",
  "language": "english",
  "speaker": "Vivian"
}
HTTP 400


# 20. Test with empty text (should fail due to NonEmptyStr validation)
POST {{base_url}}/api/v1/generate
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "text": "",
  "language": "english",
  "speaker": "Vivian"
}
HTTP 422


# 21. Test with remote provider (should fail - not implemented)
POST {{base_url}}/api/v1/generate
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model_id": "some-remote-model",
  "text": "This should fail",
  "language": "english",
  "speaker": "Vivian",
  "provider": "remote"
}
HTTP 400
[Asserts]
jsonpath "$.detail" == "Remote models not implemented yet."


# 23. Verify base64 encoding of wavs
POST {{base_url}}/api/v1/generate
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "text": "Short test.",
  "language": "english",
  "speaker": "Vivian"
}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.wavs[0]" matches /^[A-Za-z0-9+\/]+=*$/
jsonpath "$.sampling_rate" isInteger


# 24. Test with mixed language content (model should handle based on language param)
POST {{base_url}}/api/v1/generate
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "text": "Testing with mixed content: 你好 and hello",
  "language": "auto",
  "speaker": "Dylan"
}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.wavs[0]" exists
jsonpath "$.sampling_rate" == 24000


# 25. Test long text generation
POST {{base_url}}/api/v1/generate
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "text": "This is a longer text to test the generation capabilities of the text-to-speech system. It should handle multiple sentences and longer paragraphs without any issues. The system should produce high-quality audio output that maintains consistency throughout the entire generation process.",
  "language": "english",
  "speaker": "Serena"
}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.wavs[0]" exists
jsonpath "$.sampling_rate" == 24000


# =============================================
# Voice Design Model Tests (qwen3-tts-voice-design)
# =============================================

# 26. Basic voice design generation
POST {{base_url}}/api/v1/generate
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model_id": "qwen3-tts-voice-design",
  "text": "Hello, this is a test of voice design.",
  "language": "english",
  "speaker": "generated",
  "instruct": "A young female with an energetic and cheerful tone"
}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.wavs" isCollection
jsonpath "$.wavs[0]" exists
jsonpath "$.sampling_rate" == 24000


# 27. Voice design with different voice characteristics
POST {{base_url}}/api/v1/generate
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model_id": "qwen3-tts-voice-design",
  "text": "Welcome to our professional service.",
  "language": "english",
  "speaker": "generated",
  "instruct": "A deep male voice, calm and authoritative"
}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.wavs[0]" exists
jsonpath "$.sampling_rate" == 24000


# 28. Voice design without instruct (should fail)
POST {{base_url}}/api/v1/generate
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model_id": "qwen3-tts-voice-design",
  "text": "This should fail without instruct.",
  "language": "english",
  "speaker": "generated"
}
HTTP 400
[Asserts]
jsonpath "$.detail" contains "Instruct is required"


# 29. Voice design with invalid speaker (should fail - must be "generated")
POST {{base_url}}/api/v1/generate
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model_id": "qwen3-tts-voice-design",
  "text": "This should fail with invalid speaker.",
  "language": "english",
  "speaker": "Vivian",
  "instruct": "A cheerful voice"
}
HTTP 400
[Asserts]
jsonpath "$.detail" contains "Unsupported speaker"


# 30. Voice design in Chinese
POST {{base_url}}/api/v1/generate
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model_id": "qwen3-tts-voice-design",
  "text": "你好，这是语音设计测试。",
  "language": "chinese",
  "speaker": "generated",
  "instruct": "温柔的女性声音"
}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.wavs[0]" exists


# 31. Voice design in Japanese
POST {{base_url}}/api/v1/generate
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model_id": "qwen3-tts-voice-design",
  "text": "こんにちは、音声デザインのテストです。",
  "language": "japanese",
  "speaker": "generated",
  "instruct": "明るく元気な若い女性の声"
}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.wavs[0]" exists


# 32. Voice design with sampling parameters
POST {{base_url}}/api/v1/generate
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model_id": "qwen3-tts-voice-design",
  "text": "Testing with custom sampling parameters.",
  "language": "english",
  "speaker": "generated",
  "instruct": "A warm and friendly voice",
  "top_k": 50,
  "top_p": 0.9,
  "temperature": 0.7
}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.wavs[0]" exists


# =============================================
# Authentication Tests
# =============================================

# 33. Test without authentication (should fail)
POST {{base_url}}/api/v1/generate
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "text": "This should fail without auth.",
  "language": "english",
  "speaker": "Vivian"
}
HTTP 401


# 34. Test with invalid token (should fail)
POST {{base_url}}/api/v1/generate
Authorization: Bearer invalid-token-here
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "text": "This should fail with invalid token.",
  "language": "english",
  "speaker": "Vivian"
}
HTTP 401
