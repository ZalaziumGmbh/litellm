import io
import json
import wave
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Final
from unittest.mock import MagicMock

import httpx
import pytest
import respx
from pydantic import ValidationError

import litellm
from litellm.llms.azure.audio_transcription.transformation import (
    AzureSpeechAudioTranscriptionConfig,
    AzureSpeechAudioTranscriptionException,
)
from litellm.llms.base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
    BaseAudioTranscriptionConfig,
)
from litellm.types.utils import TranscriptionResponse
from litellm.utils import ProviderConfigManager


def test_azure_speech_audio_transcription_config_installed():
    config = ProviderConfigManager.get_provider_audio_transcription_config(
        model="speech/azure-stt",
        provider=litellm.LlmProviders.AZURE,
    )

    assert isinstance(config, BaseAudioTranscriptionConfig)
    assert isinstance(config, AzureSpeechAudioTranscriptionConfig)


def test_azure_speech_audio_transcription_builds_stt_url_from_cognitive_endpoint():
    config = AzureSpeechAudioTranscriptionConfig()

    url = config.get_complete_url(
        api_base="https://eastus.api.cognitive.microsoft.com/",
        api_key="test-key",
        model="speech/azure-stt",
        optional_params={"language": "fr-FR", "response_format": "verbose_json"},
        litellm_params={},
    )

    assert (
        url
        == "https://eastus.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1?language=fr-FR&format=detailed"
    )


def test_azure_speech_audio_transcription_accepts_stt_endpoint_base():
    config = AzureSpeechAudioTranscriptionConfig()

    url = config.get_complete_url(
        api_base="https://westus.stt.speech.microsoft.com",
        api_key="test-key",
        model="speech/azure-stt",
        optional_params={},
        litellm_params={},
    )

    assert (
        url
        == "https://westus.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1?language=en-US&format=simple"
    )


def test_azure_speech_audio_transcription_uses_dedicated_api_base_env(monkeypatch):
    config = AzureSpeechAudioTranscriptionConfig()

    monkeypatch.setattr(
        "litellm.llms.azure.audio_transcription.transformation.get_secret_str",
        lambda key: (
            "https://centralus.api.cognitive.microsoft.com"
            if key == "AZURE_SPEECH_API_BASE"
            else None
        ),
    )

    url = config.get_complete_url(
        api_base=None,
        api_key="test-key",
        model="speech/azure-stt",
        optional_params={},
        litellm_params={},
    )

    assert (
        url
        == "https://centralus.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1?language=en-US&format=simple"
    )


def test_azure_speech_audio_transcription_rejects_azure_openai_endpoint():
    config = AzureSpeechAudioTranscriptionConfig()

    with pytest.raises(
        AzureSpeechAudioTranscriptionException,
        match="not an Azure OpenAI endpoint",
    ):
        config.get_complete_url(
            api_base="https://example.openai.azure.com",
            api_key="test-key",
            model="speech/azure-stt",
            optional_params={},
            litellm_params={},
        )


def test_azure_speech_audio_transcription_validate_environment():
    config = AzureSpeechAudioTranscriptionConfig()

    headers = config.validate_environment(
        headers={},
        model="speech/azure-stt",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="test-key",
    )

    assert headers["Ocp-Apim-Subscription-Key"] == "test-key"
    assert headers["Content-Type"] == "audio/wav"
    assert headers["Accept"] == "application/json"


def test_azure_speech_audio_transcription_uses_dedicated_api_key_env(monkeypatch):
    config = AzureSpeechAudioTranscriptionConfig()

    monkeypatch.setattr(
        "litellm.llms.azure.audio_transcription.transformation.get_secret_str",
        lambda key: "speech-key" if key == "AZURE_SPEECH_API_KEY" else None,
    )

    headers = config.validate_environment(
        headers={},
        model="speech/azure-stt",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key=None,
    )

    assert headers["Ocp-Apim-Subscription-Key"] == "speech-key"


def test_azure_speech_audio_transcription_request_transform():
    config = AzureSpeechAudioTranscriptionConfig()
    audio = io.BytesIO(b"RIFF....WAVE")

    request_data = config.transform_audio_transcription_request(
        model="speech/azure-stt",
        audio_file=audio,
        optional_params={},
        litellm_params={},
    )

    assert isinstance(request_data, AudioTranscriptionRequestData)
    assert request_data.data == b"RIFF....WAVE"
    assert request_data.files is None
    assert request_data.content_type == "audio/wav"


@pytest.mark.parametrize(
    "payload,expected_text",
    [
        ({"DisplayText": "hello world"}, "hello world"),
        (
            {
                "RecognitionStatus": "Success",
                "NBest": [{"Display": "best text", "Confidence": 0.91}],
            },
            "best text",
        ),
    ],
)
def test_azure_speech_audio_transcription_response_transform(payload, expected_text):
    config = AzureSpeechAudioTranscriptionConfig()
    response = httpx.Response(200, json=payload)

    result = config.transform_audio_transcription_response(response)

    assert isinstance(result, TranscriptionResponse)
    assert result.text == expected_text
    assert result._hidden_params == payload


def test_azure_speech_audio_transcription_response_raises_on_failed_status():
    config = AzureSpeechAudioTranscriptionConfig()
    response = httpx.Response(
        200,
        json={
            "RecognitionStatus": "NoMatch",
            "Offset": 0,
            "Duration": 0,
        },
    )

    with pytest.raises(
        AzureSpeechAudioTranscriptionException,
        match="RecognitionStatus=NoMatch",
    ):
        config.transform_audio_transcription_response(response)


def test_azure_speech_transcription_routes_through_provider_config(monkeypatch):
    expected = TranscriptionResponse(text="hello")
    audio_handler = MagicMock(return_value=expected)

    monkeypatch.setattr(
        litellm.main.base_llm_http_handler,
        "audio_transcriptions",
        audio_handler,
    )

    response = litellm.transcription(
        model="azure/speech/azure-stt",
        file=io.BytesIO(b"RIFF....WAVE"),
        api_base="https://eastus.api.cognitive.microsoft.com",
        api_key="test-key",
        language="en-US",
    )

    assert response is expected
    audio_handler.assert_called_once()
    assert isinstance(
        audio_handler.call_args.kwargs["provider_config"],
        AzureSpeechAudioTranscriptionConfig,
    )
    assert audio_handler.call_args.kwargs["custom_llm_provider"] == "azure"


@pytest.mark.asyncio
@pytest.mark.parametrize("sample_rate,language", [(8000, "de"), (24000, "de-DE")])
async def test_fast_transcription_router_preserves_wav_and_uses_eu_multipart(
    respx_mock: respx.MockRouter, sample_rate: int, language: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    audio: Final = io.BytesIO()
    with wave.open(audio, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * sample_rate)
    audio_bytes: Final = audio.getvalue()
    endpoint: Final = respx_mock.post(
        "https://westeurope.api.cognitive.microsoft.com/speechtotext/transcriptions:transcribe",
        params={"api-version": "2025-10-15"},
    ).respond(
        200,
        json={
            "combinedPhrases": [{"text": "Guten Tag."}, {"text": "Wie geht es Ihnen?"}],
            "durationMilliseconds": 1000,
        },
    )
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "azure-stt",
                "litellm_params": {
                    "model": "azure/speech/azure-stt-fast",
                    "api_base": "https://westeurope.api.cognitive.microsoft.com",
                    "api_key": "test-speech-key",
                },
            }
        ],
        num_retries=0,
    )

    response: Final = await router.atranscription(
        model="azure-stt",
        file=("utterance.wav", audio_bytes, "audio/wav"),
        language=language,
        extra_headers={"Content-Type": "application/json"},
    )

    assert response.text == "Guten Tag. Wie geht es Ihnen?"
    assert response["duration"] == 1.0
    assert endpoint.call_count == 1
    request: Final = endpoint.calls[0].request
    assert request.headers["Ocp-Apim-Subscription-Key"] == "test-speech-key"
    assert request.headers["content-type"].startswith("multipart/form-data; boundary=")
    multipart: Final = BytesParser(policy=policy.default).parsebytes(
        f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode() + request.content
    )
    parts: Final = {part.get_param("name", header="content-disposition"): part for part in multipart.iter_parts()}
    assert json.loads(parts["definition"].get_payload(decode=True)) == {"locales": ["de-DE"]}
    assert parts["audio"].get_payload(decode=True) == audio_bytes
    assert parts["audio"].get_filename() == "utterance.wav"


def test_fast_transcription_without_language_and_silent_audio() -> None:
    config: Final = AzureSpeechAudioTranscriptionConfig(use_fast_transcription=True)
    request: Final = config.transform_audio_transcription_request(
        model="speech/azure-stt-fast", audio_file=b"RIFF....WAVE", optional_params={}, litellm_params={}
    )
    assert request.data == {"definition": "{}"}
    response: Final = config.transform_audio_transcription_response(
        httpx.Response(200, json={"combinedPhrases": [], "durationMilliseconds": 1250})
    )
    assert response.text == ""
    assert response["duration"] == 1.25
    with pytest.raises(ValidationError):
        config.transform_audio_transcription_response(httpx.Response(200, json={"error": "invalid audio"}))


@pytest.mark.parametrize(
    "filename", ["model_prices_and_context_window.json", "litellm/model_prices_and_context_window_backup.json"]
)
def test_fast_transcription_pricing_is_registered(filename: str) -> None:
    pricing: Final = json.loads((Path(__file__).parents[4] / filename).read_text())
    entry: Final = pricing["azure/speech/azure-stt-fast"]
    assert entry["input_cost_per_second"] == pytest.approx(0.36 / 3600)
    assert entry["mode"] == "audio_transcription"
    assert entry["audio_transcription_config"] == "azure_speech"
