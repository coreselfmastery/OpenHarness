"""Provider-backed perception for OpenHarness channels.

Small async clients for services that turn media into text (today: speech
transcription via Groq's OpenAI-compatible audio endpoint). They live here so
channels never import a launcher that does not exist.
"""
