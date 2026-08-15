import base64
from io import BytesIO
import json
import time
from typing import Any
import wave # <-- Added for converting raw audio arrays to WAV format
import torch
import numpy as np
import httpx  # already installed as a dependency of the `openai` package
from PIL import Image
from openai import OpenAI
from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_content_part_param import ChatCompletionContentPartParam
from comfy_api.latest import io, ui
from .iotypes import ParamClient, ParamHistory, ParamOptions, HistoryPayload, OptionsPayload


def comfy_image_to_base64_png_url(image: torch.Tensor) -> str:
    # Taken from the SaveImage ComfyUI node, convert the tensor into a regular image
    i = np.multiply(255., image.cpu().numpy())
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    # Encode the image as PNG in base64 format
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    b64_png = base64.b64encode(buffer.getvalue())
    # Return the formated string URL
    return f"data:image/png;base64,{b64_png.decode('utf-8')}"


def comfy_audio_to_base64_wav(audio: dict[str, Any]) -> tuple[str, str]:
    # ComfyUI audio format is a dictionary: {"waveform": torch.Tensor, "sample_rate": int}
    waveform = audio.get("waveform")
    sample_rate = audio.get("sample_rate", 44100)

    if waveform is None:
        raise ValueError("Provided audio input contains no waveform.")

    # Take the first channel/batch element if shape is [batch, channels, samples]
    if len(waveform.shape) == 3:
        waveform = waveform[0]

    # Clamp float values to [-1.0, 1.0] and convert to 16-bit PCM
    waveform_np = waveform.cpu().numpy()
    waveform_np = np.clip(waveform_np, -1.0, 1.0)
    waveform_np = (waveform_np * 32767.0).astype(np.int16)

    # Transpose [channels, samples] -> [samples, channels] for writing wave frames
    if len(waveform_np.shape) == 2:
        waveform_np = waveform_np.T

    # Write to an in-memory buffer using Python's wave library
    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        n_channels = waveform_np.shape[1] if len(waveform_np.shape) > 1 else 1
        wav_file.setnchannels(n_channels)
        wav_file.setsampwidth(2) # 2 bytes = 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(waveform_np.tobytes())

    b64_wav = base64.b64encode(buffer.getvalue())
    return b64_wav.decode("utf-8"), "wav"


def format_usage(usage: CompletionUsage | None) -> str | None:
    if usage is None:
        return None

    # Prompt tokens
    text = f"Prompt tokens: {usage.prompt_tokens}"
    if usage.prompt_tokens_details is not None:
        text += f" ("
        details = False

        if usage.prompt_tokens_details.audio_tokens is not None and \
            usage.prompt_tokens_details.audio_tokens > 0:
            # if details:
            # text += ", "
            text += f"audio: {usage.prompt_tokens_details.audio_tokens}"
            details = True

        if usage.prompt_tokens_details.cached_tokens is not None and \
            usage.prompt_tokens_details.cached_tokens > 0:
            if details:
                text += ", "
            text += f"cached: {usage.prompt_tokens_details.cached_tokens}"
        text += ")"

    # Completion tokens
    text += f"\nCompletions tokens: {usage.completion_tokens}"
    if usage.completion_tokens_details is not None:
        text += f" ("
        details = False

        if usage.completion_tokens_details.audio_tokens is not None and \
            usage.completion_tokens_details.audio_tokens > 0:
            text += f"audio: {usage.completion_tokens_details.audio_tokens}"
            details = True

        if usage.completion_tokens_details.reasoning_tokens is not None and \
            usage.completion_tokens_details.reasoning_tokens > 0:
            if details:
                text += ", "
            text += f"reasoning: {usage.completion_tokens_details.reasoning_tokens}"
            details = True

        if usage.completion_tokens_details.accepted_prediction_tokens is not None and \
            usage.completion_tokens_details.accepted_prediction_tokens > 0:
            if details:
                text += ", "
            text += f"prediction accepted: {usage.completion_tokens_details.accepted_prediction_tokens}"
            details = True

        if usage.completion_tokens_details.rejected_prediction_tokens is not None and \
            usage.completion_tokens_details.rejected_prediction_tokens > 0:
            if details:
                text += ", "
            text += f"prediction rejected: {usage.completion_tokens_details.rejected_prediction_tokens}"
            details = True

        if usage.completion_tokens_details.audio_tokens is not None and \
            usage.completion_tokens_details.audio_tokens > 0:
            if details:
                text += ", "
            text += f"audio: {usage.completion_tokens_details.audio_tokens}"
            # details = True
        text += ")"

    # Return the formatted text
    return text


class ChatCompletion(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="OAIAPI_ChatCompletion",
            display_name="OpenAI API - Chat Completion",
            category="OpenAI API",
            description="Generates text responses using OpenAI's chat completion API. Be sure to indicate a Vision/Audio Language Model if you are using image or audio inputs.",
            inputs=[
                ParamClient.Input(
                    id="client",
                    display_name="API Client",
                    tooltip="The OpenAI API client to use to perform the request"
                ),
                io.String.Input(
                    id="model",
                    display_name="Model",
                    tooltip="The model to use for generating text",
                    placeholder="Model name",
                ),
                io.Boolean.Input(
                    id="force_regen",
                    display_name="Force Regen",
                    tooltip="Set to true to always request a new text generation even if no widget input values have changed (no cache)",
                    default=False,
                ),
                io.Boolean.Input(
                    id="unload_after",
                    display_name="Unload Model After Generating",
                    tooltip="llama.cpp router mode: POST /models/unload to the server after generation to free VRAM/RAM",
                    default=False,
                ),
                io.String.Input(
                    id="prompt",
                    display_name="Prompt",
                    tooltip="The prompt to use for generating text",
                    multiline=True,
                    placeholder="user prompt is mandatory",
                ),
                io.String.Input(
                    id="system_prompt",
                    display_name="System Prompt",
                    optional=True,
                    tooltip="The system prompt to send along with the user prompt",
                    multiline=True,
                    placeholder="system/developer prompt is optional",
                ),
                ParamHistory.Input(
                    id="history",
                    display_name="History",
                    optional=True,
                    tooltip="Previous conversation history",
                ),
                ParamOptions.Input(
                    id="options",
                    display_name="Options",
                    optional=True,
                    tooltip="Additional options to pass with the request",
                ),
                io.Image.Input(
                    id="images",
                    display_name="image(s)",
                    optional=True,
                    tooltip="Image(s) to include in the request",
                ),
                io.Audio.Input(
                    id="audio",
                    display_name="audio",
                    optional=True,
                    tooltip="Audio file to include in the request",
                ),
            ],
            outputs=[
                io.String.Output(
                    id="response",
                    display_name="Response",
                    tooltip="Generated text response",
                ),
                ParamHistory.Output(
                    id="complete_chatcompletion",
                    display_name="History",
                    tooltip="Conversation history",
                ),
            ],
        )

    @classmethod
    def validate_inputs(cls, model: str, prompt: str) -> bool | str:
        if model == "":
            return "model must be specified"
        if prompt == "":
            return "prompt must be specified"
        return True

    @classmethod
    def fingerprint_inputs(cls, **kwargs) -> str:
        if kwargs.get("force_regen"):
            return str(time.time()) # Use timestamp for always refresh
        else:
            # Return a sorted key sorted JSON string of the inputs for fingerprinting
            # Remove force_regen as it will always be False in this path
            kwargs.pop("force_regen")
            return json.dumps(kwargs, sort_keys=True, separators=(',', ':'))

    @classmethod
    def execute(cls,
                client: OpenAI,
                model: str,
                prompt: str,
                system_prompt: str | None = None,
                history: HistoryPayload | None = None,
                options: OptionsPayload | None = None,
                images: list[torch.Tensor] | None = None,
                audio: dict[str, Any] | None = None,
                force_regen: bool = False,
                unload_after: bool = False,
                ) -> io.NodeOutput:
        # Handle options
        seed: int | None = None
        temperature: float | None = None
        max_tokens: int | None = None
        top_p: float | None = None
        frequency_penalty: float | None = None
        presence_penalty: float | None = None
        use_developer_role: bool = False
        extra_body: dict[str, Any] = {}

        if options is not None:
            extra_body = options.get_options_copy()
            if "seed" in extra_body:
                seed = extra_body["seed"]
                del extra_body["seed"]
            if "temperature" in extra_body:
                temperature = extra_body["temperature"]
                del extra_body["temperature"]
            if "max_tokens" in extra_body:
                max_tokens = extra_body["max_tokens"]
                del extra_body["max_tokens"]
            if "top_p" in extra_body:
                top_p = extra_body["top_p"]
                del extra_body["top_p"]
            if "frequency_penalty" in extra_body:
                frequency_penalty = extra_body["frequency_penalty"]
                del extra_body["frequency_penalty"]
            if "presence_penalty" in extra_body:
                presence_penalty = extra_body["presence_penalty"]
                del extra_body["presence_penalty"]
            if "use_developer_role" in extra_body:
                use_developer_role = extra_body["use_developer_role"]
                del extra_body["use_developer_role"]

        # Handle system prompt
        if history is not None:
            messages = history.get_msgs_copy()
            if system_prompt is not None and system_prompt != "":
                # Should we insert it at the beginning or replace the existing system message?
                first_msg_role = messages[0].get('role')
                if first_msg_role == "system" or first_msg_role == "developer":
                    # Replace the existing system message
                    if use_developer_role:
                        messages[0] = {
                            "role": "developer", # need literal for type hint check
                            "content": system_prompt,
                        }
                    else:
                        messages[0] = {
                            "role": "system", # need literal for type hint check
                            "content": system_prompt,
                        }
                else:
                    # insert a new system/dev message at the begining of the list
                    if use_developer_role:
                        messages.insert(0, {
                            "role": "developer", # need literal for type hint check
                            "content": system_prompt,
                        })
                    else:
                        messages.insert(0, {
                            "role": "system", # need literal for type hint check
                            "content": system_prompt,
                        })
        else:
            messages: list[ChatCompletionMessageParam] = []
            if system_prompt:
                if use_developer_role:
                    messages.append({
                        "role": "developer", # need literal for type hint check
                        "content": system_prompt,
                    })
                else:
                    messages.append({
                        "role": "system", # need literal for type hint check
                        "content": system_prompt,
                    })

        # Handle user message (supporting text, images, and audio multimodally)
        if images is not None or audio is not None:
            content: list[ChatCompletionContentPartParam] = []

            # 1. Add user prompt text
            if prompt:
                content.append(
                    {
                        "type": "text",
                        "text": prompt
                    }
                )

            # 2. Add images if any
            if images is not None:
                for image in images:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": comfy_image_to_base64_png_url(image)
                            }
                        }
                    )

            # 3. Add audio if any
            if audio is not None:
                b64_audio, audio_format = comfy_audio_to_base64_wav(audio)
                content.append(
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": b64_audio,
                            "format": audio_format
                        }
                    }
                )

            messages.append(
                {
                    "role": "user",
                    "content": content,
                }
            )
        else:
            messages.append(
                {
                    "role": "user",
                    "content": prompt
                }
            )

        # Create the completion
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            seed=seed, # deprecated, should we remove it?
            temperature=temperature,
            # should be max_completion_tokens but only vLLM has implemented it so far, Ollama and TGI have not
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            extra_body=extra_body,
            n=1
        )

        # Unload the model after generation (llama.cpp router mode)
        if unload_after:
            try:
                # The router's model-management routes live at the server root,
                # not under /v1, so strip /v1 from the OpenAI base URL.
                base = str(client.base_url).rstrip("/")
                if base.endswith("/v1"):
                    base = base[:-3]
                headers = {}
                if client.api_key and client.api_key != "-":
                    headers["Authorization"] = f"Bearer {client.api_key}"
                r = httpx.post(f"{base}/models/unload", json={"model": model}, headers=headers, timeout=60)
                if r.status_code != 200:
                    # fallback in case your build exposes the route under /v1
                    r = httpx.post(f"{base}/v1/models/unload", json={"model": model}, headers=headers, timeout=60)
                print(f"[OpenAI API] unload '{model}': HTTP {r.status_code} {r.text[:200]}")
            except Exception as e:
                # Never let an unload failure break the workflow
                print(f"[OpenAI API] failed to unload '{model}': {e}")

        # Add the response to the history
        messages.append(
            {
                "role": completion.choices[0].message.role,
                "content": completion.choices[0].message.content
            }
        )

        # Handle usage stats as text preview
        stats = format_usage(completion.usage)

        # add it to the console following the openai http call log for now as previewtext does not work yet
        print(stats)

        # Return the response and the history and the stats for the UI
        return io.NodeOutput(
            completion.choices[0].message.content,
            HistoryPayload(messages),
            ui=ui.PreviewText(stats) if stats else ui.PreviewText(""),
        )