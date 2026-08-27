import os
import re
import json
import base64
import mimetypes
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
                        "default": "http://127.0.0.1:11434",
                    },
                ),

                "ollama_path": (
                    "STRING",
                    {
                        "default": "/opt/ollama/ollama",
                    },
                ),

                "timeout": (
                    "INT",
                    {
                        "default": 600,
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
    # 主函数
    # =========================================================

    def chat(
            self,
            prompt,
            model,
            cache_model="是",
            ollama_url="http://127.0.0.1:11434",
            ollama_path="/opt/ollama/ollama",
            timeout=600,
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
        print(f"Ollama Path : {ollama_path}")

        # =====================================================
        # 查找 Prompt 中的媒体文件
        # =====================================================

        media_paths = self.extract_media_paths(prompt)

        if media_paths:

            print()
            print("Detected media files:")

            for path in media_paths:

                if os.path.isfile(path):

                    size = os.path.getsize(path)

                    print(
                        f"  {path} "
                        f"({size / 1024 / 1024:.2f} MB)"
                    )

                else:

                    print(
                        f"  [NOT FOUND] {path}"
                    )

        # =====================================================
        # 构建 messages
        # =====================================================

        message = {
            "role": "user",
            "content": prompt,
        }

        # =====================================================
        # 对图片进行 Ollama Vision API 处理
        #
        # 注意：
        #
        # Ollama /api/chat 对 vision 模型通常要求：
        #
        # {
        #   "role": "user",
        #   "content": "...",
        #   "images": ["base64..."]
        # }
        #
        # 如果 Qwen3.8 模型支持 vision，则把图片
        # 自动转换为 base64。
        # =====================================================

        image_data = []

        for path in media_paths:

            if not os.path.isfile(path):
                continue

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
                        f"[Image] Added: {path}"
                    )

                except Exception as e:

                    print(
                        f"[Image] Failed: "
                        f"{path} -> {e}"
                    )

        if image_data:

            message["images"] = image_data

        # =====================================================
        # 请求 JSON
        # =====================================================

        request_data = {
            "model": model,

            "messages": [
                message
            ],

            "stream": False,

            "keep_alive": (
                -1
                if cache_model == "是"
                else 0
            ),
        }

        # =====================================================
        # 打印请求信息
        # =====================================================

        print()
        print("Sending request to Ollama...")

        print(
            f"Endpoint: "
            f"{ollama_url}/api/chat"
        )

        print(
            f"Images: "
            f"{len(image_data)}"
        )

        print(
            f"Keep Alive: "
            f"{request_data['keep_alive']}"
        )

        # =====================================================
        # 调用 /api/chat
        # =====================================================

        try:

            response = self.http_post(
                f"{ollama_url}/api/chat",
                request_data,
                timeout,
            )

            # =================================================
            # 解析 JSON
            # =================================================

            if not isinstance(
                    response,
                    dict
            ):

                return (
                    "Invalid Ollama response:\n"
                    + str(response),
                )

            # =================================================
            # Ollama API Error
            # =================================================

            if "error" in response:

                error = response.get(
                    "error",
                    "Unknown Ollama error"
                )

                print(
                    f"[Ollama Error] {error}"
                )

                return (
                    f"Ollama API error:\n"
                    f"{error}",
                )

            # =================================================
            # 获取 message.content
            # =================================================

            message_data = response.get(
                "message",
                {}
            )

            if not isinstance(
                    message_data,
                    dict
            ):

                message_data = {}

            result = message_data.get(
                "content",
                ""
            )

            if result is None:

                result = ""

            result = str(result).strip()

            # =================================================
            # 清理最终文本
            #
            # HTTP API 正常情况下不会存在 ANSI，
            # 这里作为保险。
            # =================================================

            result = self.clean_result(
                result
            )

            # =================================================
            # 输出统计信息
            # =================================================

            print()
            print(
                f"Response length: "
                f"{len(result)}"
            )

            if "done" in response:

                print(
                    f"Done: "
                    f"{response.get('done')}"
                )

            if "total_duration" in response:

                duration = (
                        response["total_duration"]
                        / 1_000_000_000
                )

                print(
                    f"Total duration: "
                    f"{duration:.2f}s"
                )

            if "eval_count" in response:

                print(
                    f"Eval tokens: "
                    f"{response['eval_count']}"
                )

            # =================================================
            # cache_model = 否
            #
            # keep_alive=0 已经要求 Ollama 立即卸载。
            #
            # 这里再执行一次 stop 作为保险。
            # =================================================

            if cache_model == "否":

                print()
                print(
                    f"[Cache] Stopping model: "
                    f"{model}"
                )

                self.stop_model(
                    ollama_url,
                    model,
                    ollama_path,
                )

            else:

                print()
                print(
                    f"[Cache] Keep model loaded: "
                    f"{model}"
                )

            print()
            print("=" * 70)
            print("Ollama Finished")
            print("=" * 70)

            return (
                result,
            )

        except urllib.error.HTTPError as e:

            try:

                error_body = e.read().decode(
                    "utf-8",
                    errors="replace"
                )

            except Exception:

                error_body = str(e)

            print(
                f"[HTTP Error] "
                f"{e.code}: "
                f"{error_body}"
            )

            if cache_model == "否":

                self.stop_model(
                    ollama_url,
                    model,
                    ollama_path,
                )

            return (
                f"Ollama HTTP error "
                f"{e.code}:\n"
                f"{error_body}",
            )

        except urllib.error.URLError as e:

            print(
                f"[Connection Error] "
                f"{e}"
            )

            if cache_model == "否":

                self.stop_model(
                    ollama_url,
                    model,
                    ollama_path,
                )

            return (
                "Cannot connect to Ollama:\n"
                f"{ollama_url}\n\n"
                f"{e}",
            )

        except TimeoutError:

            print(
                "[ERROR] Ollama request timeout"
            )

            if cache_model == "否":

                self.stop_model(
                    ollama_url,
                    model,
                    ollama_path,
                )

            return (
                f"Ollama request timeout "
                f"after {timeout} seconds.",
            )

        except Exception as e:

            print(
                "[Ollama Exception]",
                repr(e)
            )

            if cache_model == "否":

                self.stop_model(
                    ollama_url,
                    model,
                    ollama_path,
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

            response_body = response.read()

        text = response_body.decode(
            "utf-8",
            errors="replace"
        )

        return json.loads(
            text
        )

    # =========================================================
    # Stop Model
    # =========================================================

    @staticmethod
    def stop_model(
            ollama_url,
            model,
            ollama_path,
    ):

        # -----------------------------------------------------
        # 第一种方式：
        #
        # Ollama API:
        #
        # /api/generate
        #
        # keep_alive=0
        #
        # 让模型立即卸载。
        # -----------------------------------------------------

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
                "[Ollama Stop API] "
                f"Model released: {model}"
            )

            return True

        except Exception as e:

            print(
                "[Ollama Stop API] Failed:",
                repr(e)
            )

        # -----------------------------------------------------
        # 第二种方式：
        #
        # 如果 API 方式失败，
        # 使用本地 ollama stop。
        # -----------------------------------------------------

        try:

            if not ollama_path:

                return False

            env = os.environ.copy()

            env["TERM"] = "dumb"

            env["NO_COLOR"] = "1"

            env["CI"] = "1"

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

                env=env,
            )

            if result.returncode == 0:

                print(
                    "[Ollama Stop CLI] "
                    f"Model stopped: {model}"
                )

                return True

            print(
                "[Ollama Stop CLI] "
                f"Failed: {result.stdout}"
            )

        except Exception as e:

            print(
                "[Ollama Stop CLI] "
                "Failed:",
                repr(e)
            )

        return False

    # =========================================================
    # 判断图片
    # =========================================================

    @staticmethod
    def is_image(path):

        image_extensions = (
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".bmp",
            ".gif",
        )

        return path.lower().endswith(
            image_extensions
        )

    # =========================================================
    # 判断视频
    # =========================================================

    @staticmethod
    def is_video(path):

        video_extensions = (
            ".mp4",
            ".mov",
            ".mkv",
            ".avi",
            ".webm",
            ".m4v",
        )

        return path.lower().endswith(
            video_extensions
        )

    # =========================================================
    # 提取媒体路径
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
        #
        # 支持：
        #
        # /root/test/a.png
        #
        # "/root/test/a.png"
        #
        # '/root/test/a.png'
        # =====================================================

        linux_pattern = re.compile(
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

        # =====================================================
        # Windows path
        # =====================================================

        windows_pattern = re.compile(
            r"""
            (?:
                "([^"]+)"
                |
                '([^']+)'
                |
                ([A-Za-z]:[\\/][^\s"'<>]+)
            )
            """,
            re.VERBOSE
        )

        candidates = []

        for match in linux_pattern.findall(
                prompt
        ):

            candidates.extend(
                [
                    x
                    for x in match
                    if x
                ]
            )

        for match in windows_pattern.findall(
                prompt
        ):

            candidates.extend(
                [
                    x
                    for x in match
                    if x
                ]
            )

        # =====================================================
        # 去重 + 验证
        # =====================================================

        for path in candidates:

            path = path.strip()

            path = path.rstrip(
                ".,;:!?，。；：！？）》）"
            )

            if not path.lower().endswith(
                    extensions
            ):
                continue

            # -------------------------------------------------
            # 这里只返回存在的文件
            # -------------------------------------------------

            if os.path.isfile(path):

                real_path = os.path.abspath(
                    path
                )

                if real_path not in paths:

                    paths.append(
                        real_path
                    )

        return paths

    # =========================================================
    # 清理最终结果
    # =========================================================

    @staticmethod
    def clean_result(text):

        if not text:
            return ""

        # -----------------------------------------------------
        # ANSI
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
        # ASCII control chars
        # -----------------------------------------------------

        text = re.sub(
            r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]",
            "",
            text
        )

        # -----------------------------------------------------
        # spinner
        # -----------------------------------------------------

        text = re.sub(
            r"[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]",
            "",
            text
        )

        # -----------------------------------------------------
        # Ollama CLI 日志
        # -----------------------------------------------------

        text = re.sub(
            r"Added image\s+'.*?'",
            "",
            text,
            flags=re.IGNORECASE
        )

        text = re.sub(
            r"Added video\s+'.*?'",
            "",
            text,
            flags=re.IGNORECASE
        )

        # -----------------------------------------------------
        # 清理 CR
        # -----------------------------------------------------

        text = text.replace(
            "\r",
            ""
        )

        return text.strip()


# =============================================================
# 独立 Ollama Stop 节点
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

        ollama_url = (
            ollama_url.rstrip("/")
        )

        print()
        print("=" * 70)
        print("Ollama Stop Model")
        print("=" * 70)

        print(
            f"Model : {model}"
        )

        print(
            f"URL   : {ollama_url}"
        )

        success = (
            OllamaPathChat.stop_model(
                ollama_url,
                model,
                ollama_path,
            )
        )

        if success:

            status = (
                f"Successfully stopped "
                f"model: {model}"
            )

        else:

            status = (
                f"Failed to stop model: "
                f"{model}"
            )

        print(status)

        return (
            status,
        )

