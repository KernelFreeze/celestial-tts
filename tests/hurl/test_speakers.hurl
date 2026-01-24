# Test Custom Speaker Creation and Usage
# Run with: hurl --test test_speakers.hurl
# Note: Requires the server to be running on 127.0.0.1:8080

# 1. Try to create a speaker with a model that doesn't support custom speakers (should fail)
POST http://127.0.0.1:8080/speakers
Content-Type: application/json
{
  "model_id": "qwen3-tts-preset",
  "name": "Test Speaker",
  "text": "Hello, this is a test voice.",
  "language": "en",
  "instruct": "A friendly, warm voice"
}
HTTP 400
[Asserts]
jsonpath "$.detail" contains "does not support custom speakers"


# 2. Create a custom speaker with qwen3-tts-custom model
POST http://127.0.0.1:8080/speakers
Content-Type: application/json
{
  "model_id": "qwen3-tts-custom",
  "name": "Friendly Voice",
  "text": "Hello, this is a test of my custom voice. I sound friendly and warm.",
  "language": "en",
  "instruct": "A friendly, warm, and cheerful voice"
}
HTTP 200
[Captures]
speaker_id: jsonpath "$.speaker.id"
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.speaker.name" == "Friendly Voice"
jsonpath "$.speaker.id" exists
jsonpath "$.speaker.created_at" exists


# 3. Test with invalid language (should fail)
POST http://127.0.0.1:8080/speakers
Content-Type: application/json
{
  "model_id": "qwen3-tts-custom",
  "name": "Invalid Language Test",
  "text": "Test text",
  "language": "invalid_lang",
  "instruct": "Test voice"
}
HTTP 400
[Asserts]
jsonpath "$.detail" contains "Unsupported language"


# 4. Get all speakers for qwen3-tts-custom model
GET http://127.0.0.1:8080/speakers?model_id=qwen3-tts-custom
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.speakers" isCollection
jsonpath "$.speakers[*].id" includes {{speaker_id}}
jsonpath "$.speakers[?(@.id == '{{speaker_id}}')].name" == "Friendly Voice"


# 5. Get speakers for preset model (should return empty list)
GET http://127.0.0.1:8080/speakers?model_id=qwen3-tts-preset
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.speakers" isEmpty


# 6. Use the created custom speaker to generate audio
POST http://127.0.0.1:8080/generate
Content-Type: application/json
{
  "model_id": "qwen3-tts-custom",
  "text": "This is a test using my custom voice!",
  "language": "en",
  "speaker": "{{speaker_id}}",
  "instruct": "Speaking with enthusiasm"
}
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.wavs" isCollection
jsonpath "$.wavs[0]" exists
jsonpath "$.sampling_rate" exists


# 7. Try to use invalid speaker UUID in generation (should fail)
POST http://127.0.0.1:8080/generate
Content-Type: application/json
{
  "model_id": "qwen3-tts-custom",
  "text": "This should fail",
  "language": "en",
  "speaker": "00000000-0000-0000-0000-000000000000",
  "instruct": "Test"
}
HTTP 400
[Asserts]
jsonpath "$.detail" contains "Unsupported speaker"


# 8. Create another speaker with multiple text samples
POST http://127.0.0.1:8080/speakers
Content-Type: application/json
{
  "model_id": "qwen3-tts-custom",
  "name": "Professional Voice",
  "text": ["Welcome to our service.", "Thank you for choosing us.", "We appreciate your business."],
  "language": "en",
  "instruct": ["Professional", "Clear", "Confident"]
}
HTTP 200
[Captures]
speaker_id_2: jsonpath "$.speaker.id"
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.speaker.name" == "Professional Voice"


# 9. Verify both speakers are returned in the list
GET http://127.0.0.1:8080/speakers?model_id=qwen3-tts-custom
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.speakers" count >= 2
jsonpath "$.speakers[*].id" includes {{speaker_id}}
jsonpath "$.speakers[*].id" includes {{speaker_id_2}}


# 10. Test multilingual speaker creation (Japanese)
POST http://127.0.0.1:8080/speakers
Content-Type: application/json
{
  "model_id": "qwen3-tts-custom",
  "name": "Japanese Voice",
  "text": "こんにちは、これはテストです。",
  "language": "ja",
  "instruct": "優しい声"
}
HTTP 200
[Captures]
speaker_id_jp: jsonpath "$.speaker.id"
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.speaker.name" == "Japanese Voice"


# 11. Try to delete speaker with invalid UUID format (should fail)
DELETE http://127.0.0.1:8080/speakers/invalid-uuid?model_id=qwen3-tts-custom
HTTP 400
[Asserts]
jsonpath "$.detail" == "Invalid speaker UUID"


# 12. Try to delete non-existent speaker (should fail)
DELETE http://127.0.0.1:8080/speakers/00000000-0000-0000-0000-000000000000?model_id=qwen3-tts-custom
HTTP 404
[Asserts]
jsonpath "$.detail" == "Speaker not found"


# 13. Try to delete speaker from model that doesn't support custom speakers (should fail)
DELETE http://127.0.0.1:8080/speakers/{{speaker_id}}?model_id=qwen3-tts-preset
HTTP 400
[Asserts]
jsonpath "$.detail" contains "does not support custom speakers"


# 14. Delete the Japanese speaker successfully
DELETE http://127.0.0.1:8080/speakers/{{speaker_id_jp}}?model_id=qwen3-tts-custom
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.message" == "Speaker deleted successfully"


# 15. Verify the Japanese speaker is no longer in the list
GET http://127.0.0.1:8080/speakers?model_id=qwen3-tts-custom
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
jsonpath "$.speakers[*].id" not includes {{speaker_id_jp}}


# 16. Try to use the deleted speaker for generation (should fail)
POST http://127.0.0.1:8080/generate
Content-Type: application/json
{
  "model_id": "qwen3-tts-custom",
  "text": "This should fail",
  "language": "en",
  "speaker": "{{speaker_id_jp}}",
  "instruct": "Test"
}
HTTP 400
[Asserts]
jsonpath "$.detail" contains "Unsupported speaker"


# 17. Delete the first speaker
DELETE http://127.0.0.1:8080/speakers/{{speaker_id}}?model_id=qwen3-tts-custom
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"


# 18. Delete the second speaker
DELETE http://127.0.0.1:8080/speakers/{{speaker_id_2}}?model_id=qwen3-tts-custom
HTTP 200
[Asserts]
jsonpath "$.status" == "ok"
