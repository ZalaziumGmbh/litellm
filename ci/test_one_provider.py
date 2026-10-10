"""Offline adapter, endpoint and invoice-rate checks against the built image."""
import json
import os
import unittest
from unittest.mock import patch

import httpx
import litellm
from litellm.types.llms.openai import HttpxBinaryResponseContent
from litellm.llms.azure.audio_transcription.transformation import AzureSpeechAudioTranscriptionConfig
from litellm.llms.azure.text_to_speech.transformation import AzureAVATextToSpeechConfig
from litellm.llms.vertex_ai.text_to_speech.transformation import VertexAITextToSpeechConfig
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
from litellm.utils import ProviderConfigManager


class ProviderTests(unittest.TestCase):
    def test_fast_transcription_selection_payload_and_duration(self):
        adapter = ProviderConfigManager.get_provider_audio_transcription_config(
            'speech/azure-stt-fast', litellm.LlmProviders.AZURE)
        self.assertIsInstance(adapter, AzureSpeechAudioTranscriptionConfig)
        self.assertTrue(adapter.use_fast_transcription)
        url = adapter.get_complete_url('https://swedencentral.api.cognitive.microsoft.com',
            'synthetic', 'speech/azure-stt-fast', {}, {})
        self.assertEqual(url, 'https://swedencentral.api.cognitive.microsoft.com/speechtotext/transcriptions:transcribe?api-version=2025-10-15')
        data = adapter.transform_audio_transcription_request('speech/azure-stt-fast',
            ('test.wav', b'RIFFsynthetic', 'audio/wav'), {'language': 'de'}, {})
        self.assertEqual(json.loads(data.data['definition']), {'locales': ['de-DE']})
        result = adapter.transform_audio_transcription_response(httpx.Response(200,
            json={'combinedPhrases': [{'text': 'Synthetic test.'}], 'durationMilliseconds': 1250}))
        self.assertEqual(result.text, 'Synthetic test.')
        self.assertEqual(result.duration, 1.25)

    def test_voice_slots_and_native_voices(self):
        for adapter, native in [(AzureAVATextToSpeechConfig(), 'de-DE-KatjaNeural'),
                                (VertexAITextToSpeechConfig(), 'de-DE-Neural2-G')]:
            self.assertEqual(adapter.resolve_voice_alias('F1', {'F1': native}), native)
            self.assertEqual(adapter.resolve_voice_alias(native, {'F1': native}), native)
            with self.assertRaises(ValueError):
                adapter.resolve_voice_alias('F1', {'F1': 123})

    def test_explicit_region_cannot_silently_move(self):
        with patch.dict(litellm.model_cost, {'vertex_ai/synthetic-region': {'supported_regions': ['us-east5']}}):
            with self.assertRaisesRegex(ValueError, 'cross-region'):
                VertexBase.get_vertex_region('eu', 'synthetic-region')
        with patch.dict(litellm.model_cost, {'vertex_ai/synthetic-region': {'supported_regions': ['eu']}}):
            self.assertEqual(VertexBase.get_vertex_region('eu', 'synthetic-region'), 'eu')

    def test_google_endpoint_override_survives_dispatch(self):
        endpoint = 'https://eu-texttospeech.googleapis.com/v1/text:synthesize'
        response = HttpxBinaryResponseContent(httpx.Response(200, content=b'RIFFsynthetic'))
        with patch.object(VertexAITextToSpeechConfig, 'dispatch_text_to_speech', return_value=response) as dispatch:
            litellm.speech(model='vertex_ai/neural2', input='Synthetic speech.', voice='de-DE-Neural2-G',
                api_base=endpoint, vertex_project='synthetic', vertex_location='eu',
                vertex_credentials='{}', response_format='wav')
        self.assertEqual(dispatch.call_args.kwargs['api_base'], endpoint)

    def test_audio_rates_have_no_reselling_multiplier(self):
        for provider, name, price in [('azure', 'synthetic-azure-tts', 0.000015),
                                      ('vertex_ai', 'synthetic-google-tts', 0.000016)]:
            litellm.register_model({provider + '/' + name: {
                'litellm_provider': provider, 'mode': 'audio_speech',
                'input_cost_per_character': price, 'output_cost_per_character': 0.0}})
            cost = litellm.completion_cost(model=provider + '/' + name,
                custom_llm_provider=provider, call_type='speech', prompt='a' * 1000)
            self.assertAlmostEqual(cost, price * 1000, places=10)


if __name__ == '__main__':
    if os.getuid() == 0:
        raise RuntimeError('Run as tester')
    unittest.main()
