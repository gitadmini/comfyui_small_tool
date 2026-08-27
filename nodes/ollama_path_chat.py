import os
import re
import json
import base64
import subprocess
import urllib.request
import urllib.error


class OllamaPathChat:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "dynamicPrompts": False,
                    },
                ),

                "model": (
                    "STRING",
                    {
                        "default": "qwen3.8:27b",
                    },
                ),

                "cache_model": (
                    ["是", "否"],
                    {
                        "default": "是",
                    },
                ),
            },

            "optional": {
                "ollama_url": (
                    "STRING",
                    {
                        "default":
                            "http://127.0.0.1:11434",
                    },
                ),

                "ollama_path": (
                    "STRING",
                    {
                        "default":
                            "/opt/ollama/ollama",
                    },
                ),

                "timeout": (
                    "INT",
                    {
                        "default": 1800,
                        "min": 10,
                        "max": 7200,
                        "step": 10,
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)

    RETURN_NAMES = ("result",)

    FUNCTION = "chat"

    CATEGORY = "Ollama/Qwen3.8"

    OUTPUT_NODE = False

    # =========================================================
    # Chat
    # =========================================================

    def chat(
            self,
            prompt,
            model,
            cache_model="是",
            ollama_url="http://127.0.0.1:11434",
            ollama_path="/opt/ollama/ollama",
            timeout=1800,
    ):

        if not prompt or not prompt.strip():
            return ("",)

        ollama_url = ollama_url.rstrip("/")

        print()
        print("=" * 70)
        print("Ollama Qwen3.8 HTTP API")
        print("=" * 70)

        print(f"Model       : {model}")
        print(f"Cache Model : {cache_model}")
        print(f"Ollama URL  : {ollama_url}")

        # =====================================================
        # 查找 Prompt 中的图片
        # =====================================================

        media_paths = self.extract_media_paths(prompt)

        image_data = []

        if media_paths:

            print()
            print("Detected files:")

            for path in media_paths:

                print(f"  {path}")

                if not os.path.isfile(path):

                    print(
                        "  [WARNING] File not found"
                    )

                    continue

                # -------------------------------------------------
                # 图片
                # -------------------------------------------------

                if self.is_image(path):

                    try:

                        with open(
                                path,
                                "rb"
                        ) as f:

                            encoded = base64.b64encode(
                                f.read()
                            ).decode("utf-8")

                        image_data.append(
                            encoded
                        )

                        print(
                            "  [Image] Added"
                        )

                    except Exception as e:

                        print(
                            f"  [Image] Failed: {e}"
                        )

                # -------------------------------------------------
                # 视频
                # -------------------------------------------------

                elif self.is_video(path):

                    print(
                        "  [Video] Detected"
                    )

                    print(
                        "  [Video] "
                        "Standard Ollama /api/chat "
                        "does not directly accept video."
                    )

        # =====================================================
        # Message
        # =====================================================

        message = {
            "role": "user",
            "content": prompt,
        }

        # =====================================================
        # Vision
        # =====================================================

        if image_data:

            message["images"] = image_data

            print(
                f"Images sent to Ollama: "
                f"{len(image_data)}"
            )

        # =====================================================
        # keep_alive
        #
        # 是  -> -1
        # 否  -> 0
        # =====================================================

        if cache_model == "是":

            keep_alive = -1

        else:

            keep_alive = 0

        # =====================================================
        # Request
        # =====================================================

        request_data = {
            "model": model,

            "messages": [
                message
            ],

            "stream": False,

            "keep_alive": keep_alive,
        }

        print()
        print(
            "POST "
            f"{ollama_url}/api/chat"
        )

        print(
            f"keep_alive = {keep_alive}"
        )

        # =====================================================
        # API
        # =====================================================

        try:

            response = self.http_post(
                f"{ollama_url}/api/chat",
                request_data,
                timeout,
            )

            # =================================================
            # API error
            # =================================================

            if not isinstance(
                    response,
                    dict
            ):

                return (
                    "Invalid Ollama API response.",
                )

            if "error" in response:

                error = response.get(
                    "error",
                    "Unknown error"
                )

                print(
                    f"[Ollama API Error] "
                    f"{error}"
                )

                return (
                    f"Ollama API error:\n"
                    f"{error}",
                )

            # =================================================
            # 获取 message
            # =================================================

            message_response = response.get(
                "message"
            )

            if not isinstance(
                    message_response,
                    dict
            ):

                print(
                    "[ERROR] "
                    "message field not found"
                )

                return (
                    "Ollama response does not "
                    "contain message.",
                )

            # =================================================
            # 只获取 content
            #
            # 注意：
            #
            # 不读取：
            #
            # message["thinking"]
            #
            # 只读取：
            #
            # message["content"]
            # =================================================

            result = message_response.get(
                "content",
                ""
            )

            if result is None:

                result = ""

            result = str(
                result
            )

            # =================================================
            # 最终清理
            # =================================================

            result = self.clean_result(
                result
            )

            print()
            print(
                f"Result length: "
                f"{len(result)}"
            )

            # =================================================
            # Token / 时间
            # =================================================

            if "total_duration" in response:

                total_seconds = (
                        response["total_duration"]
                        / 1_000_000_000
                )

                print(
                    f"Total duration: "
                    f"{total_seconds:.2f}s"
                )

            if "load_duration" in response:

                load_seconds = (
                        response["load_duration"]
                        / 1_000_000_000
                )

                print(
                    f"Load duration: "
                    f"{load_seconds:.2f}s"
                )

            if "prompt_eval_count" in response:

                print(
                    f"Prompt tokens: "
                    f"{response['prompt_eval_count']}"
                )

            if "eval_count" in response:

                print(
                    f"Output tokens: "
                    f"{response['eval_count']}"
                )

            # =================================================
            # Cache
            # =================================================

            if cache_model == "是":

                print(
                    f"[Cache] "
                    f"Model kept loaded: {model}"
                )

            else:

                print(
                    f"[Cache] "
                    f"Model will be released: {model}"
                )

            print()
            print("=" * 70)
            print("Ollama Finished")
            print("=" * 70)

            return (
                result,
            )

        # =====================================================
        # HTTP Error
        # =====================================================

        except urllib.error.HTTPError as e:

            try:

                error_body = (
                    e.read()
                        .decode(
                        "utf-8",
                        errors="replace"
                    )
                )

            except Exception:

                error_body = str(e)

            print(
                f"[HTTP {e.code}] "
                f"{error_body}"
            )

            return (
                f"Ollama HTTP error "
                f"{e.code}:\n"
                f"{error_body}",
            )

        # =====================================================
        # Connection Error
        # =====================================================

        except urllib.error.URLError as e:

            print(
                f"[Connection Error] "
                f"{e}"
            )

            return (
                "Cannot connect to Ollama:\n"
                f"{ollama_url}\n\n"
                f"{e}",
            )

        # =====================================================
        # Timeout
        # =====================================================

        except TimeoutError:

            print(
                "[ERROR] "
                "Ollama request timeout"
            )

            return (
                f"Ollama request timeout "
                f"after {timeout} seconds.",
            )

        # =====================================================
        # Other
        # =====================================================

        except Exception as e:

            print(
                "[Ollama Exception]",
                repr(e)
            )

            return (
                f"Ollama error:\n"
                f"{type(e).__name__}: {e}",
            )

    # =========================================================
    # HTTP POST
    # =========================================================

    @staticmethod
    def http_post(
            url,
            data,
            timeout,
    ):

        body = json.dumps(
            data,
            ensure_ascii=False
        ).encode("utf-8")

        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
        )

        request.add_header(
            "Content-Type",
            "application/json"
        )

        request.add_header(
            "Accept",
            "application/json"
        )

        with urllib.request.urlopen(
                request,
                timeout=timeout
        ) as response:

            response_body = (
                response.read()
            )

        text = response_body.decode(
            "utf-8",
            errors="replace"
        )

        return json.loads(
            text
        )

    # =========================================================
    # 提取文件路径
    # =========================================================

    @staticmethod
    def extract_media_paths(prompt):

        extensions = (
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".bmp",
            ".gif",
            ".mp4",
            ".mov",
            ".mkv",
            ".avi",
            ".webm",
            ".m4v",
        )

        paths = []

        # =====================================================
        # Linux path
        # =====================================================

        pattern = re.compile(
            r"""
            (?:
                "([^"]+)"
                |
                '([^']+)'
                |
                (/[^\s"'<>]+)
            )
            """,
            re.VERBOSE
        )

        for match in pattern.findall(
                prompt
        ):

            candidates = [
                x
                for x in match
                if x
            ]

            for path in candidates:

                path = path.strip()

                path = path.rstrip(
                    ".,;:!?，。；：！？）》）"
                )

                if not path.lower().endswith(
                        extensions
                ):
                    continue

                if os.path.isfile(path):

                    path = os.path.abspath(
                        path
                    )

                    if path not in paths:

                        paths.append(
                            path
                        )

        return paths

    # =========================================================
    # Image
    # =========================================================

    @staticmethod
    def is_image(path):

        return path.lower().endswith(
            (
                ".png",
                ".jpg",
                ".jpeg",
                ".webp",
                ".bmp",
                ".gif",
            )
        )

    # =========================================================
    # Video
    # =========================================================

    @staticmethod
    def is_video(path):

        return path.lower().endswith(
            (
                ".mp4",
                ".mov",
                ".mkv",
                ".avi",
                ".webm",
                ".m4v",
            )
        )

    # =========================================================
    # Clean result
    # =========================================================

    @staticmethod
    def clean_result(text):

        if not text:

            return ""

        # -----------------------------------------------------
        # ANSI ESC
        # -----------------------------------------------------

        text = re.sub(
            r"\x1b\[[0-9;?]*[ -/]*[@-~]",
            "",
            text
        )

        text = re.sub(
            r"\x1b\][^\x07]*(?:\x07|\x1b\\)",
            "",
            text
        )

        text = text.replace(
            "\x1b",
            ""
        )

        # -----------------------------------------------------
        # 控制字符
        # -----------------------------------------------------

        text = re.sub(
            r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]",
            "",
            text
        )

        # -----------------------------------------------------
        # CR
        # -----------------------------------------------------

        text = text.replace(
            "\r",
            ""
        )

        return text.strip()


# =============================================================
# Ollama Stop Model
# =============================================================

class OllamaStopModel:

    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {

                "model": (
                    "STRING",
                    {
                        "default":
                            "qwen3.8:27b",
                    },
                ),
            },

            "optional": {

                "ollama_url": (
                    "STRING",
                    {
                        "default":
                            "http://127.0.0.1:11434",
                    },
                ),

                "ollama_path": (
                    "STRING",
                    {
                        "default":
                            "/opt/ollama/ollama",
                    },
                ),
            },
        }

    RETURN_TYPES = (
        "STRING",
    )

    RETURN_NAMES = (
        "status",
    )

    FUNCTION = "stop"

    CATEGORY = "Ollama/Qwen3.8"

    OUTPUT_NODE = False

    def stop(
            self,
            model,
            ollama_url="http://127.0.0.1:11434",
            ollama_path="/opt/ollama/ollama",
    ):

        ollama_url = ollama_url.rstrip("/")

        print()
        print("=" * 70)
        print("Ollama Stop Model")
        print("=" * 70)

        print(
            f"Model: {model}"
        )

        # =====================================================
        # 首选 HTTP API
        #
        # /api/generate
        # keep_alive = 0
        # =====================================================

        try:

            data = {
                "model": model,
                "keep_alive": 0,
            }

            response = (
                OllamaPathChat.http_post(
                    f"{ollama_url}/api/generate",
                    data,
                    60,
                )
            )

            print(
                f"[Ollama API] "
                f"Model released: {model}"
            )

            return (
                f"Successfully stopped: {model}",
            )

        except Exception as e:

            print(
                "[Ollama API Stop Failed]",
                repr(e)
            )

        # =====================================================
        # API 失败后使用 CLI
        # =====================================================

        try:

            if (
                    not ollama_path
                    or not os.path.exists(ollama_path)
            ):

                return (
                    f"Failed to stop model: "
                    f"{model}",
                )

            result = subprocess.run(
                [
                    ollama_path,
                    "stop",
                    model,
                ],

                stdout=subprocess.PIPE,

                stderr=subprocess.STDOUT,

                text=True,

                encoding="utf-8",

                errors="replace",

                timeout=60,
            )

            if result.returncode == 0:

                return (
                    f"Successfully stopped: "
                    f"{model}",
                )

            return (
                f"Failed to stop model:\n"
                f"{result.stdout}",
            )

        except Exception as e:

            return (
                f"Stop model error:\n"
                f"{e}",
            )

