# Azure and Google speech through an EU LiteLLM proxy

[eu_proxy_config.yaml](eu_proxy_config.yaml) exposes four selectable models through the existing OpenAI-compatible audio endpoints. Merge its `model_list` entries into the proxy configuration and run a build containing Azure Fast Transcription and configurable voice mappings

| Model alias | Endpoint | Provider service |
| --- | --- | --- |
| `azure-stt` | `/v1/audio/transcriptions` | Azure Speech Fast Transcription, West Europe |
| `azure-tts` | `/v1/audio/speech` | Azure Speech Neural TTS, West Europe |
| `google-stt` | `/v1/audio/transcriptions` | Google Speech-to-Text V2 Chirp 3, EU |
| `google-tts` | `/v1/audio/speech` | Google Chirp 3 HD TTS, EU |

## Credentials and activation

Create an Azure Speech resource in **West Europe** using the **Standard S0 pay-as-you-go tier**, and supply its key as `AZURE_SPEECH_API_KEY` in the proxy's secret environment. An Azure OpenAI key cannot replace the Speech resource key

For Google, enable billing, `speech.googleapis.com`, and `texttospeech.googleapis.com` in the project. Set `GOOGLE_SPEECH_PROJECT` to that project ID and `GOOGLE_SPEECH_CREDENTIALS` to the path of a service-account JSON file mounted read-only inside the proxy container. Grant Speech Client for recognition and Service Usage Consumer on the billing project, and follow Google's [TTS authentication setup](https://cloud.google.com/text-to-speech/docs/authentication). When using application default credentials or workload identity instead, omit `vertex_credentials`

Rebuild and restart the proxy with the updated code and configuration. If a virtual key has a model allowlist, include the four aliases. Verify that `/model/info` exposes `audio_transcription` and `audio_speech` modes before selecting the desired STT and TTS models in the voicebot. These entries do not change existing model selections or add automatic fallbacks

## Audio and voices

The integrations accept the voicebot's mono PCM WAV recordings, including 8 kHz phone audio and 24 kHz browser audio. Transcription returns a JSON object with `text`. A short language code such as `de` maps to `de-DE`; omitting the language enables provider language detection

Use `response_format: wav` for speech synthesis. The configurable `voice_mappings` keep the existing F1–F5 and M1–M5 slots usable, with German provider voices. These slots have different voices on each provider. Provider-native voice names remain accepted; choose a name with the desired language prefix for other languages

These are HTTP requests per completed utterance. They do not provide partial streaming transcription. Keep Google synchronous recognition clips at or below its 60-second / 10 MB limits. Use `json` or `text` for transcription response formats; word timestamps and `verbose_json` are not provided by this example

## EU processing

Both Azure endpoints are fixed to West Europe. Google's recognizer location and STT endpoint are both `eu`. The full Google TTS `api_base` is essential: setting `vertex_location` alone does not replace that adapter's global TTS endpoint

Keep the proxy, audio storage, logs, tracing services, and any fallback models in the EU as well. This configuration controls speech API routing; it does not by itself establish residency for every service used by the application. Confirm the cloud account's contractual requirements before production use

Provider references: [Azure regions](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/regions), [Azure Fast Transcription](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/fast-transcription-create), [Google STT locations](https://cloud.google.com/speech-to-text/v2/docs/locations), [Google TTS regional endpoints](https://cloud.google.com/text-to-speech/docs/endpoints)

## Live smoke test

Set `LITELLM_BASE_URL` to the proxy origin and `LITELLM_API_KEY` to a virtual key using your secret manager. Run these commands in a shell with a short German `sample.wav` file. Successful STT calls return `{"text":"..."}`; successful TTS calls produce playable WAV files

```bash
for model in azure-stt google-stt; do
  curl --fail-with-body "$LITELLM_BASE_URL/v1/audio/transcriptions" \
    -H "Authorization: Bearer $LITELLM_API_KEY" \
    -F "model=$model" -F "language=de" -F "file=@sample.wav;type=audio/wav"
done

for model in azure-tts google-tts; do
  curl --fail-with-body "$LITELLM_BASE_URL/v1/audio/speech" \
    -H "Authorization: Bearer $LITELLM_API_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$model\",\"input\":\"Guten Tag, wie kann ich Ihnen helfen?\",\"voice\":\"F1\",\"response_format\":\"wav\"}" \
    --output "$model.wav"
done
```

## Cost accounting

The example uses USD list-price estimates before discounts, taxes, free allowances, or application markup: Azure Fast STT $0.36/hour, Azure Neural TTS $15/million characters, Google STT $0.016/minute at the first paid tier, and Google Chirp 3 HD TTS $30/million characters. Adjust `model_info` for your billing policy. Provider billing is authoritative; retries and billing increments can differ from the proxy estimate

Azure Fast STT's West Europe rate was checked against the [Azure Retail Prices API](https://prices.azure.com/api/retail/prices) on 2026-10-08. See [Azure Speech pricing](https://azure.microsoft.com/en-us/pricing/details/speech/), [Google STT pricing](https://cloud.google.com/speech-to-text/pricing), and [Google TTS pricing](https://cloud.google.com/text-to-speech/pricing)
