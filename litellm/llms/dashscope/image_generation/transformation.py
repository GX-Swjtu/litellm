"""
DashScope Image Generation Configuration

Handles transformation between OpenAI-compatible format and DashScope multimodal-generation API.

API endpoint: POST https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation

Request format:
{
    "model": "qwen-image-2.0-pro",
    "input": {
        "messages": [{"role": "user", "content": [{"text": "<prompt>"}]}]
    },
    "parameters": {"size": "1024*1024", ...}
}

Response format:
{
    "output": {
        "choices": [{"message": {"content": [{"image": "<url>"}]}}]
    },
    "usage": {"input_tokens": 0, "output_tokens": 0, "width": 1024, "height": 1024, "image_count": 1}
}
"""

import base64
import re
from typing import TYPE_CHECKING, Any, Final
from urllib.parse import urlparse

import httpx

from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIImageGenerationOptionalParams,
)
from litellm.types.utils import ImageObject, ImageResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

DEFAULT_API_BASE: Final = "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
MAX_DASHSCOPE_IMAGE_BYTES: Final = 25 * 1024 * 1024
_DASHSCOPE_RESULT_HOST: Final = re.compile(
    r"^dashscope-result(?:-[a-z0-9-]+)?\.oss-"
    r"(?:cn|ap|eu|us|me|na)-[a-z0-9-]+\.aliyuncs\.com$"
)
_PNG_SIGNATURE: Final = b"\x89PNG\r\n\x1a\n"

# Maps OpenAI size strings (WxH) to DashScope size strings (W*H)
OPENAI_TO_DASHSCOPE_SIZE: Final[dict] = {
    "256x256": "256*256",
    "512x512": "512*512",
    "1024x1024": "1024*1024",
    "1792x1024": "1792*1024",
    "1024x1792": "1024*1792",
    "2048x2048": "2048*2048",
}


class DashScopeImageGenerationConfig(BaseImageGenerationConfig):
    """
    Configuration for DashScope image generation (qwen-image-2.0, qwen-image-2.0-pro).
    """

    def get_supported_openai_params(self, model: str) -> list[OpenAIImageGenerationOptionalParams]:
        return ["n", "response_format", "size"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_params: Final = self.get_supported_openai_params(model)
        mapped: Final[dict] = {}
        for k, v in non_default_params.items():
            if k in optional_params:
                continue
            if k not in supported_params:
                continue
            if k == "size":
                # Convert "WxH" → "W*H"
                mapped["size"] = OPENAI_TO_DASHSCOPE_SIZE.get(v, v.replace("x", "*"))
            elif k == "n":
                mapped["n"] = v
            elif k == "response_format":
                # DashScope always returns a temporary OSS URL. Keep this as an
                # internal response-conversion instruction and do not forward it.
                mapped["response_format"] = v
        return mapped

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        return api_base or get_secret_str("DASHSCOPE_API_BASE_IMAGE") or DEFAULT_API_BASE

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        final_api_key: Final = api_key or get_secret_str("DASHSCOPE_API_KEY")
        if not final_api_key:
            raise ValueError("DASHSCOPE_API_KEY is not set")
        headers["Authorization"] = f"Bearer {final_api_key}"
        headers["Content-Type"] = "application/json"
        return headers

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform OpenAI-style image generation request to DashScope multimodal-generation format.
        """
        parameters: Final[dict] = {}
        for k, v in optional_params.items():
            if k == "response_format":
                continue
            parameters[k] = v

        return {
            "model": model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": prompt}],
                    }
                ]
            },
            "parameters": parameters,
        }

    @staticmethod
    def _validate_result_url(image_url: str) -> None:
        parsed = urlparse(image_url)
        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError("invalid port") from exc

        hostname = (parsed.hostname or "").lower().rstrip(".")
        if (
            parsed.scheme != "https"
            or parsed.username is not None
            or parsed.password is not None
            or port not in (None, 443)
            or parsed.fragment
            or not _DASHSCOPE_RESULT_HOST.fullmatch(hostname)
        ):
            raise ValueError("untrusted DashScope result URL")

    @classmethod
    def _download_result_as_base64(cls, image_url: str) -> str:
        cls._validate_result_url(image_url)
        image_bytes = bytearray()
        timeout = httpx.Timeout(120.0, connect=10.0)
        with httpx.Client(
            follow_redirects=False,
            timeout=timeout,
            trust_env=True,
        ) as client:
            with client.stream(
                "GET",
                image_url,
                headers={"Accept": "image/png"},
            ) as response:
                if response.status_code != 200:
                    raise ValueError(f"DashScope result download returned HTTP {response.status_code}")

                content_type = response.headers.get("content-type", "")
                media_type = content_type.split(";", 1)[0].strip().lower()
                if media_type not in {"image/png", "application/octet-stream"}:
                    raise ValueError("DashScope result is not a PNG response")

                content_length = response.headers.get("content-length")
                if content_length:
                    try:
                        declared_length = int(content_length)
                    except ValueError as exc:
                        raise ValueError("invalid DashScope result content length") from exc
                    if declared_length > MAX_DASHSCOPE_IMAGE_BYTES:
                        raise ValueError("DashScope result exceeds the image size limit")

                for chunk in response.iter_bytes(chunk_size=64 * 1024):
                    image_bytes.extend(chunk)
                    if len(image_bytes) > MAX_DASHSCOPE_IMAGE_BYTES:
                        raise ValueError("DashScope result exceeds the image size limit")

        if not image_bytes.startswith(_PNG_SIGNATURE):
            raise ValueError("DashScope result does not contain PNG data")
        return base64.b64encode(image_bytes).decode("ascii")

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ImageResponse:
        """
        Transform DashScope response to litellm ImageResponse.

        DashScope response: output.choices[0].message.content[0].image
        OpenAI response:    data[0].url
        """
        if raw_response.status_code != 200:
            raise self.get_error_class(
                error_message=raw_response.text,
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        try:
            response_data: Final = raw_response.json()
        except Exception as e:
            raise self.get_error_class(
                error_message=f"Failed to parse DashScope image generation response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        # DashScope can return API-level errors in a 200 response body.
        # Example: {"code": "InvalidParameter", "message": "Size not supported"}
        if "code" in response_data and "output" not in response_data:
            raise self.get_error_class(
                error_message=str(response_data.get("message", response_data)),
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        if not model_response.data:
            model_response.data = []

        choices: Final = response_data.get("output", {}).get("choices", [])
        for choice in choices:
            content_list = choice.get("message", {}).get("content", [])
            for content_item in content_list:
                image_url = content_item.get("image")
                if image_url:
                    if optional_params.get("response_format") == "b64_json":
                        try:
                            encoded_image = self._download_result_as_base64(image_url)
                        except (httpx.HTTPError, ValueError) as exc:
                            raise self.get_error_class(
                                error_message=(f"Failed to retrieve the generated DashScope image: {exc}"),
                                status_code=502,
                                headers=raw_response.headers,
                            ) from exc
                        model_response.data.append(ImageObject(b64_json=encoded_image))
                    else:
                        model_response.data.append(ImageObject(url=image_url))

        return model_response
