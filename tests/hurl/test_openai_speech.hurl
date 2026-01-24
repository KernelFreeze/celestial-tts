# Test OpenAI-Compatible Speech Endpoint
# Run with: hurl --variables-file tests/hurl/vars.env --test tests/hurl/test_openai_speech.hurl
# Note: Requires the server to be running on 127.0.0.1:8080

# =============================================
# Basic Speech Generation Tests
# =============================================

# 1. Basic speech generation with OpenAI model name and voice
POST {{base_url}}/api/v1/audio/speech
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model": "tts-1",
  "voice": "alloy",
  "input": "Hello, this is a test of the OpenAI compatible speech API."
}
HTTP 200
[Asserts]
header "Content-Type" contains "audio/"


# 2. Test with tts-1-hd model
POST {{base_url}}/api/v1/audio/speech
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model": "tts-1-hd",
  "voice": "nova",
  "input": "Testing with the high definition model."
}
HTTP 200
[Asserts]
header "Content-Type" contains "audio/"


# 3. Test with native model ID
POST {{base_url}}/api/v1/audio/speech
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model": "qwen3-tts-preset",
  "voice": "Vivian",
  "input": "Testing with native model ID and speaker name."
}
HTTP 200
[Asserts]
header "Content-Type" contains "audio/"


# =============================================
# OpenAI Voice Name Tests
# =============================================

# 4. Test OpenAI voice: alloy (maps to Vivian)
POST {{base_url}}/api/v1/audio/speech
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model": "tts-1",
  "voice": "alloy",
  "input": "Testing the alloy voice."
}
HTTP 200
[Asserts]
header "Content-Type" contains "audio/"


# 5. Test OpenAI voice: echo (maps to Dylan)
POST {{base_url}}/api/v1/audio/speech
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model": "tts-1",
  "voice": "echo",
  "input": "Testing the echo voice."
}
HTTP 200
[Asserts]
header "Content-Type" contains "audio/"


# 6. Test OpenAI voice: fable (maps to Serena)
POST {{base_url}}/api/v1/audio/speech
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model": "tts-1",
  "voice": "fable",
  "input": "Testing the fable voice."
}
HTTP 200
[Asserts]
header "Content-Type" contains "audio/"


# 7. Test OpenAI voice: onyx (maps to Eric)
POST {{base_url}}/api/v1/audio/speech
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model": "tts-1",
  "voice": "onyx",
  "input": "Testing the onyx voice."
}
HTTP 200
[Asserts]
header "Content-Type" contains "audio/"


# 8. Test OpenAI voice: nova (maps to Aiden)
POST {{base_url}}/api/v1/audio/speech
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model": "tts-1",
  "voice": "nova",
  "input": "Testing the nova voice."
}
HTTP 200
[Asserts]
header "Content-Type" contains "audio/"


# 9. Test OpenAI voice: shimmer (maps to Sohee)
POST {{base_url}}/api/v1/audio/speech
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model": "tts-1",
  "voice": "shimmer",
  "input": "Testing the shimmer voice."
}
HTTP 200
[Asserts]
header "Content-Type" contains "audio/"


# =============================================
# Native Speaker Name Tests
# =============================================

# 10. Test with native speaker: Vivian
POST {{base_url}}/api/v1/audio/speech
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model": "tts-1",
  "voice": "Vivian",
  "input": "Testing native speaker Vivian."
}
HTTP 200
[Asserts]
header "Content-Type" contains "audio/"


# 11. Test with native speaker: Ryan
POST {{base_url}}/api/v1/audio/speech
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model": "tts-1",
  "voice": "Ryan",
  "input": "Testing native speaker Ryan."
}
HTTP 200
[Asserts]
header "Content-Type" contains "audio/"


# 12. Test with native speaker: Uncle_Fu
POST {{base_url}}/api/v1/audio/speech
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model": "tts-1",
  "voice": "Uncle_Fu",
  "input": "Testing native speaker Uncle Fu."
}
HTTP 200
[Asserts]
header "Content-Type" contains "audio/"


# =============================================
# Audio Format Tests
# =============================================

# 13. Test mp3 format (default)
POST {{base_url}}/api/v1/audio/speech
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model": "tts-1",
  "voice": "alloy",
  "input": "Testing MP3 format."
}
HTTP 200
[Asserts]
header "Content-Type" == "audio/mpeg"


# 14. Test wav format
POST {{base_url}}/api/v1/audio/speech
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model": "tts-1",
  "voice": "alloy",
  "input": "Testing WAV format.",
  "response_format": "wav"
}
HTTP 200
[Asserts]
header "Content-Type" == "audio/wav"


# 15. Test opus format
POST {{base_url}}/api/v1/audio/speech
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model": "tts-1",
  "voice": "alloy",
  "input": "Testing OPUS format.",
  "response_format": "opus"
}
HTTP 200
[Asserts]
header "Content-Type" == "audio/opus"


# 16. Test flac format
POST {{base_url}}/api/v1/audio/speech
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model": "tts-1",
  "voice": "alloy",
  "input": "Testing FLAC format.",
  "response_format": "flac"
}
HTTP 200
[Asserts]
header "Content-Type" == "audio/flac"


# 17. Test pcm format
POST {{base_url}}/api/v1/audio/speech
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model": "tts-1",
  "voice": "alloy",
  "input": "Testing PCM format.",
  "response_format": "pcm"
}
HTTP 200
[Asserts]
header "Content-Type" == "audio/pcm"


# =============================================
# Speed Parameter Tests
# =============================================

# 18. Test with speed 0.5 (slower)
POST {{base_url}}/api/v1/audio/speech
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model": "tts-1",
  "voice": "alloy",
  "input": "Testing slower speech.",
  "speed": 0.5
}
HTTP 200
[Asserts]
header "Content-Type" contains "audio/"


# 19. Test with speed 1.0 (normal)
POST {{base_url}}/api/v1/audio/speech
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model": "tts-1",
  "voice": "alloy",
  "input": "Testing normal speech speed.",
  "speed": 1.0
}
HTTP 200
[Asserts]
header "Content-Type" contains "audio/"


# 20. Test with speed 2.0 (faster)
POST {{base_url}}/api/v1/audio/speech
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model": "tts-1",
  "voice": "alloy",
  "input": "Testing faster speech.",
  "speed": 2.0
}
HTTP 200
[Asserts]
header "Content-Type" contains "audio/"


# =============================================
# Multilingual Content Tests
# =============================================

# 21. Test with Chinese text (auto-detect)
POST {{base_url}}/api/v1/audio/speech
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model": "tts-1",
  "voice": "alloy",
  "input": "你好，这是中文测试。"
}
HTTP 200
[Asserts]
header "Content-Type" contains "audio/"


# 22. Test with Japanese text
POST {{base_url}}/api/v1/audio/speech
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model": "tts-1",
  "voice": "shimmer",
  "input": "こんにちは、これは日本語のテストです。"
}
HTTP 200
[Asserts]
header "Content-Type" contains "audio/"


# 23. Test with Korean text
POST {{base_url}}/api/v1/audio/speech
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model": "tts-1",
  "voice": "shimmer",
  "input": "안녕하세요, 이것은 한국어 테스트입니다."
}
HTTP 200
[Asserts]
header "Content-Type" contains "audio/"


# 24. Test with Spanish text
POST {{base_url}}/api/v1/audio/speech
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model": "tts-1",
  "voice": "fable",
  "input": "Hola, esta es una prueba en español."
}
HTTP 200
[Asserts]
header "Content-Type" contains "audio/"


# 25. Test with French text
POST {{base_url}}/api/v1/audio/speech
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model": "tts-1",
  "voice": "fable",
  "input": "Bonjour, ceci est un test en français."
}
HTTP 200
[Asserts]
header "Content-Type" contains "audio/"


# 26. Test with German text
POST {{base_url}}/api/v1/audio/speech
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model": "tts-1",
  "voice": "onyx",
  "input": "Hallo, das ist ein deutscher Test."
}
HTTP 200
[Asserts]
header "Content-Type" contains "audio/"


# =============================================
# Long Text Tests
# =============================================

# 27. Test with longer text
POST {{base_url}}/api/v1/audio/speech
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model": "tts-1",
  "voice": "nova",
  "input": "This is a longer text to test the speech generation capabilities. It should handle multiple sentences and longer paragraphs without any issues. The system should produce high-quality audio output that maintains consistency throughout the entire generation process. This helps ensure that the API can handle realistic use cases where users need to generate longer form audio content."
}
HTTP 200
[Asserts]
header "Content-Type" contains "audio/"


# =============================================
# Error Handling Tests
# =============================================

# 28. Test with empty input (should fail)
POST {{base_url}}/api/v1/audio/speech
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model": "tts-1",
  "voice": "alloy",
  "input": ""
}
HTTP 400
[Asserts]
jsonpath "$.detail" contains "empty"


# 29. Test with whitespace-only input (should fail)
POST {{base_url}}/api/v1/audio/speech
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model": "tts-1",
  "voice": "alloy",
  "input": "   "
}
HTTP 400
[Asserts]
jsonpath "$.detail" contains "empty"


# 30. Test with invalid voice (should fail)
POST {{base_url}}/api/v1/audio/speech
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model": "tts-1",
  "voice": "invalid_voice_name",
  "input": "This should fail."
}
HTTP 400
[Asserts]
jsonpath "$.detail" contains "Unsupported voice"


# 31. Test with unsupported format (should fail)
POST {{base_url}}/api/v1/audio/speech
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model": "tts-1",
  "voice": "alloy",
  "input": "This should fail.",
  "response_format": "mp4"
}
HTTP 400
[Asserts]
jsonpath "$.detail" contains "Unsupported response_format"


# 32. Test with speed too low (should fail)
POST {{base_url}}/api/v1/audio/speech
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model": "tts-1",
  "voice": "alloy",
  "input": "This should fail.",
  "speed": 0.1
}
HTTP 422


# 33. Test with speed too high (should fail)
POST {{base_url}}/api/v1/audio/speech
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model": "tts-1",
  "voice": "alloy",
  "input": "This should fail.",
  "speed": 5.0
}
HTTP 422


# 34. Test missing required field: model (should fail)
POST {{base_url}}/api/v1/audio/speech
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "voice": "alloy",
  "input": "This should fail."
}
HTTP 422


# 35. Test missing required field: voice (should fail)
POST {{base_url}}/api/v1/audio/speech
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model": "tts-1",
  "input": "This should fail."
}
HTTP 422


# 36. Test missing required field: input (should fail)
POST {{base_url}}/api/v1/audio/speech
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model": "tts-1",
  "voice": "alloy"
}
HTTP 422


# =============================================
# Authentication Tests
# =============================================

# 37. Test without authentication (should fail)
POST {{base_url}}/api/v1/audio/speech
Content-Type: application/json
{
  "model": "tts-1",
  "voice": "alloy",
  "input": "This should fail without auth."
}
HTTP 401


# 38. Test with invalid token (should fail)
POST {{base_url}}/api/v1/audio/speech
Authorization: Bearer invalid-token-here
Content-Type: application/json
{
  "model": "tts-1",
  "voice": "alloy",
  "input": "This should fail with invalid token."
}
HTTP 401


# =============================================
# Edge Cases and Special Characters
# =============================================

# 39. Test with special characters
POST {{base_url}}/api/v1/audio/speech
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model": "tts-1",
  "voice": "alloy",
  "input": "Testing with special characters: @#$%^&*()!? and numbers 123456."
}
HTTP 200
[Asserts]
header "Content-Type" contains "audio/"


# 40. Test with punctuation and formatting
POST {{base_url}}/api/v1/audio/speech
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model": "tts-1",
  "voice": "nova",
  "input": "Hello! How are you today? I'm fine, thank you. Let's test: semicolons; commas, and periods."
}
HTTP 200
[Asserts]
header "Content-Type" contains "audio/"


# 41. Test with quotes and apostrophes
POST {{base_url}}/api/v1/audio/speech
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model": "tts-1",
  "voice": "fable",
  "input": "She said, \"Hello!\" and I replied, \"It's a beautiful day, isn't it?\""
}
HTTP 200
[Asserts]
header "Content-Type" contains "audio/"


# 42. Test with newlines and whitespace
POST {{base_url}}/api/v1/audio/speech
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model": "tts-1",
  "voice": "echo",
  "input": "First line.\n\nSecond line with double newline.\tTab character here."
}
HTTP 200
[Asserts]
header "Content-Type" contains "audio/"


# 43. Test with emoji characters
POST {{base_url}}/api/v1/audio/speech
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model": "tts-1",
  "voice": "alloy",
  "input": "Hello! 😊 This is a test with emojis. 🎉🎊"
}
HTTP 200
[Asserts]
header "Content-Type" contains "audio/"


# =============================================
# Mixed Language and All Parameters
# =============================================

# 44. Test with all parameters specified
POST {{base_url}}/api/v1/audio/speech
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model": "tts-1-hd",
  "voice": "nova",
  "input": "Testing with all parameters specified for maximum compatibility.",
  "response_format": "wav",
  "speed": 1.25
}
HTTP 200
[Asserts]
header "Content-Type" == "audio/wav"
header "Content-Disposition" contains "attachment"
header "Content-Disposition" contains "speech.wav"


# 45. Test case insensitivity of format parameter
POST {{base_url}}/api/v1/audio/speech
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model": "tts-1",
  "voice": "alloy",
  "input": "Testing format case insensitivity.",
  "response_format": "MP3"
}
HTTP 200
[Asserts]
header "Content-Type" == "audio/mpeg"


# 46. Test mixed language content (English + Chinese)
POST {{base_url}}/api/v1/audio/speech
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model": "tts-1",
  "voice": "alloy",
  "input": "Hello, this is a test. 你好，这是一个测试。"
}
HTTP 200
[Asserts]
header "Content-Type" contains "audio/"


# 47. Test maximum speed (4.0)
POST {{base_url}}/api/v1/audio/speech
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model": "tts-1",
  "voice": "nova",
  "input": "Testing maximum speed of four point zero.",
  "speed": 4.0
}
HTTP 200
[Asserts]
header "Content-Type" contains "audio/"


# 48. Test minimum speed (0.25)
POST {{base_url}}/api/v1/audio/speech
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model": "tts-1",
  "voice": "nova",
  "input": "Testing minimum speed of zero point two five.",
  "speed": 0.25
}
HTTP 200
[Asserts]
header "Content-Type" contains "audio/"
