# Test Custom Speaker Creation (Voice Cloning) and Usage
# Run with: hurl --variables-file tests/hurl/vars.env --test tests/hurl/test_speakers.hurl
# Note: Requires the server to be running on 127.0.0.1:8080
#
# This test file uses voice design to generate reference audio, then uses that
# audio to test the voice cloning endpoint.

# 1. Generate reference audio using voice design (to use for cloning)
POST {{base_url}}/api/v1/generate
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model_id": "qwen3-tts-voice-design",
  "text": "Hello, this is a test of my custom voice. I sound friendly and warm.",
  "language": "english",
  "speaker": "generated",
  "instruct": "A friendly, warm, and cheerful female voice"
}
HTTP 200
[Captures]
reference_audio: jsonpath "$.wavs[0]"
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.wavs[0]" exists


# 2. Try to create a speaker with a model that doesn't support custom speakers (should fail)
POST {{base_url}}/api/v1/speakers
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "name": "Test Speaker",
  "text": "Hello, this is a test voice.",
  "audio": "{{reference_audio}}"
}
HTTP 400
[Asserts]
jsonpath "$.detail" contains "does not support custom speakers"


# 3. Create a custom speaker with voice cloning using the generated audio
POST {{base_url}}/api/v1/speakers
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model_id": "qwen3-tts-voice-clone",
  "name": "Friendly Voice",
  "text": "Hello, this is a test of my custom voice. I sound friendly and warm.",
  "audio": "{{reference_audio}}"
}
HTTP 200
[Captures]
speaker_id: jsonpath "$.speaker.id"
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.speaker.name" == "Friendly Voice"
jsonpath "$.speaker.id" exists


# 4. Test with invalid audio data (should fail)
POST {{base_url}}/api/v1/speakers
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model_id": "qwen3-tts-voice-clone",
  "name": "Invalid Audio Test",
  "text": "Test text",
  "audio": "not-valid-base64-or-url"
}
HTTP 400
[Asserts]
jsonpath "$.detail" contains "Audio input must be"


# 5. Get all speakers for qwen3-tts-voice-clone model
GET {{base_url}}/api/v1/speakers?model_id=qwen3-tts-voice-clone
Authorization: Bearer {{token}}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.speakers" isCollection
jsonpath "$.speakers[*].id" includes {{speaker_id}}
jsonpath "$.speakers[?(@.id == '{{speaker_id}}')].name" includes "Friendly Voice"


# 6. Get speakers for preset model (should return empty list)
GET {{base_url}}/api/v1/speakers?model_id=qwen3-tts-preset
Authorization: Bearer {{token}}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.speakers" count == 9


# 7. Use the created custom speaker to generate audio
POST {{base_url}}/api/v1/generate
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model_id": "qwen3-tts-voice-clone",
  "text": "This is a test using my cloned voice!",
  "language": "english",
  "speaker": "{{speaker_id}}"
}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.wavs" isCollection
jsonpath "$.wavs[0]" exists
jsonpath "$.sampling_rate" exists


# 8. Try to use invalid speaker UUID in generation (should fail)
POST {{base_url}}/api/v1/generate
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model_id": "qwen3-tts-voice-clone",
  "text": "This should fail",
  "language": "english",
  "speaker": "00000000-0000-0000-0000-000000000000"
}
HTTP 404
[Asserts]
jsonpath "$.detail" contains "not found"


# 9. Generate another reference audio for a second speaker
POST {{base_url}}/api/v1/generate
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model_id": "qwen3-tts-voice-design",
  "text": "Welcome to our service. Thank you for choosing us.",
  "language": "english",
  "speaker": "generated",
  "instruct": "A professional, clear, and confident male voice"
}
HTTP 200
[Captures]
reference_audio_2: jsonpath "$.wavs[0]"
[Asserts]
jsonpath "$.status" == "ok"


# 10. Create another speaker with the second reference audio
POST {{base_url}}/api/v1/speakers
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model_id": "qwen3-tts-voice-clone",
  "name": "Professional Voice",
  "text": "Welcome to our service. Thank you for choosing us.",
  "audio": "{{reference_audio_2}}"
}
HTTP 200
[Captures]
speaker_id_2: jsonpath "$.speaker.id"
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.speaker.name" == "Professional Voice"


# 11. Verify both speakers are returned in the list
GET {{base_url}}/api/v1/speakers?model_id=qwen3-tts-voice-clone
Authorization: Bearer {{token}}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.speakers" count >= 2
jsonpath "$.speakers[*].id" includes {{speaker_id}}
jsonpath "$.speakers[*].id" includes {{speaker_id_2}}


# 12. Generate Japanese reference audio for multilingual test
POST {{base_url}}/api/v1/generate
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model_id": "qwen3-tts-voice-design",
  "text": "こんにちは、これはテストです。",
  "language": "japanese",
  "speaker": "generated",
  "instruct": "優しい女性の声"
}
HTTP 200
[Captures]
reference_audio_jp: jsonpath "$.wavs[0]"
[Asserts]
jsonpath "$.status" == "ok"


# 13. Test multilingual speaker creation (Japanese)
POST {{base_url}}/api/v1/speakers
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model_id": "qwen3-tts-voice-clone",
  "name": "Japanese Voice",
  "text": "こんにちは、これはテストです。",
  "audio": "{{reference_audio_jp}}"
}
HTTP 200
[Captures]
speaker_id_jp: jsonpath "$.speaker.id"
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.speaker.name" == "Japanese Voice"


# 14. Try to delete speaker with invalid UUID format (should fail)
DELETE {{base_url}}/api/v1/speakers
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model_id": "qwen3-tts-voice-clone",
  "speaker_id": "invalid-uuid",
  "provider": "local"
}
HTTP 422

# 15. Try to delete non-existent speaker (should fail)
DELETE {{base_url}}/api/v1/speakers
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model_id": "qwen3-tts-voice-clone",
  "speaker_id": "00000000-0000-0000-0000-000000000000",
  "provider": "local"
}
HTTP 404
[Asserts]
jsonpath "$.detail" contains "not found"


# 16. Try to delete speaker from model that doesn't support custom speakers (should fail)
DELETE {{base_url}}/api/v1/speakers
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "speaker_id": "{{speaker_id}}",
  "provider": "local"
}
HTTP 400
[Asserts]
jsonpath "$.detail" contains "does not support custom speakers"


# 17. Delete the Japanese speaker successfully
DELETE {{base_url}}/api/v1/speakers
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model_id": "qwen3-tts-voice-clone",
  "speaker_id": "{{speaker_id_jp}}"
}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.message" == "Speaker deleted successfully"


# 18. Verify the Japanese speaker is no longer in the list
GET {{base_url}}/api/v1/speakers?model_id=qwen3-tts-voice-clone
Authorization: Bearer {{token}}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.speakers[*].id" not includes {{speaker_id_jp}}


# 19. Try to use the deleted speaker for generation (should fail)
POST {{base_url}}/api/v1/generate
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model_id": "qwen3-tts-voice-clone",
  "text": "This should fail",
  "language": "english",
  "speaker": "{{speaker_id_jp}}"
}
HTTP 404
[Asserts]
jsonpath "$.detail" contains "not found"


# 20. Delete the first speaker
DELETE {{base_url}}/api/v1/speakers
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model_id": "qwen3-tts-voice-clone",
  "speaker_id": "{{speaker_id}}"
}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"


# 21. Delete the second speaker
DELETE {{base_url}}/api/v1/speakers
Authorization: Bearer {{token}}
Content-Type: application/json
{
  "model_id": "qwen3-tts-voice-clone",
  "speaker_id": "{{speaker_id_2}}"
}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"


# 22. Test creating speaker without authentication (should fail)
POST {{base_url}}/api/v1/speakers
Content-Type: application/json
{
  "model_id": "qwen3-tts-voice-clone",
  "name": "Unauthorized Voice",
  "text": "This should fail without auth.",
  "audio": "{{reference_audio}}"
}
HTTP 401


# 23. Test getting speakers without authentication (should fail)
GET {{base_url}}/api/v1/speakers?model_id=qwen3-tts-voice-clone
HTTP 401


# 24. Test deleting speaker without authentication (should fail)
DELETE {{base_url}}/api/v1/speakers
Content-Type: application/json
{
  "model_id": "qwen3-tts-voice-clone",
  "speaker_id": "00000000-0000-0000-0000-000000000000"
}
HTTP 401


# 25. Test with invalid token for speaker creation (should fail)
POST {{base_url}}/api/v1/speakers
Authorization: Bearer invalid-token-here
Content-Type: application/json
{
  "model_id": "qwen3-tts-voice-clone",
  "name": "Invalid Token Voice",
  "text": "This should fail with invalid token.",
  "audio": "{{reference_audio}}"
}
HTTP 401
