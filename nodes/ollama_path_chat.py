import os
import re
import subprocess
import threading


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
                "ollama_path": (
                    "STRING",
                    {
                        "default": "/root/ollama",
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
    # Main
    # =========================================================

    def chat(
            self,
            prompt,
            model,
            cache_model="是",
            ollama_path="/root/ollama",
            timeout=600,
    ):

        if not prompt or not prompt.strip():
            return ("",)

        executable = ollama_path.strip()

        if not executable:
            executable = "ollama"

        print("\n")
        print("=" * 70)
        print("Ollama Qwen3.8 Path Chat")
        print("=" * 70)

        print(f"Model       : {model}")
        print(f"Cache Model : {cache_model}")
        print(f"Ollama      : {executable}")
        print(f"Prompt size : {len(prompt)}")

        # =====================================================
        # Detect local media paths
        # =====================================================

        media_paths = self.extract_media_paths(prompt)

        if media_paths:

            print("\nDetected media files:")

            for path in media_paths:

                print(f"  {path}")

        else:

            print("\nNo local media file detected.")

        # =====================================================
        # Check media files
        # =====================================================

        for path in media_paths:

            if not os.path.isfile(path):

                print(
                    f"[WARNING] File does not exist: {path}"
                )

            else:

                size = os.path.getsize(path)

                print(
                    f"[MEDIA] {path} "
                    f"({size / 1024 / 1024:.2f} MB)"
                )

        # =====================================================
        # Execute Ollama
        # =====================================================

        command = [
            executable,
            "run",
            model,
            prompt,
        ]

        print("\nStarting Ollama...")

        try:

            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

            output_lines = []

            # =================================================
            # Read Ollama output
            # =================================================

            def reader():

                try:

                    for line in process.stdout:

                        print(
                            "[Ollama]",
                            line.rstrip()
                        )

                        output_lines.append(line)

                except Exception as e:

                    print(
                        "[Ollama Reader Error]",
                        e
                    )

            thread = threading.Thread(
                target=reader,
                daemon=True,
            )

            thread.start()

            # =================================================
            # Wait
            # =================================================

            try:

                process.wait(
                    timeout=timeout
                )

            except subprocess.TimeoutExpired:

                print(
                    "[ERROR] Ollama timeout"
                )

                process.kill()

                # 超时也尝试释放模型
                if cache_model == "否":
                    self.unload_model(
                        executable,
                        model
                    )

                return (
                    f"Ollama timeout after "
                    f"{timeout} seconds.",
                )

            thread.join(timeout=5)

            result = "".join(
                output_lines
            ).strip()

            # =================================================
            # Ollama execution failed
            # =================================================

            if process.returncode != 0:

                print(
                    f"[ERROR] Ollama exit code: "
                    f"{process.returncode}"
                )

                # 即使执行失败，如果用户选择否，
                # 也尝试释放模型
                if cache_model == "否":

                    self.unload_model(
                        executable,
                        model
                    )

                return (
                    f"Ollama execution failed "
                    f"(exit code "
                    f"{process.returncode}):\n\n"
                    f"{result}",
                )

            # =================================================
            # Cache control
            # =================================================

            if cache_model == "是":

                # 不执行 ollama stop
                #
                # Ollama 会按照自己的 keep_alive
                # 策略继续保持模型在内存中。

                print(
                    "\n[Cache] Model cache enabled."
                )

                print(
                    f"[Cache] Keep model loaded: {model}"
                )

            else:

                print(
                    "\n[Cache] Model cache disabled."
                )

                print(
                    f"[Cache] Stopping model: {model}"
                )

                self.unload_model(
                    executable,
                    model
                )

            print("\n")
            print("=" * 70)
            print("Ollama Finished")
            print("=" * 70)

            return (
                result,
            )

        except FileNotFoundError:

            return (
                "Cannot find Ollama executable:\n"
                f"{executable}\n\n"
                "Please check ollama_path.",
            )

        except Exception as e:

            # 发生异常时，如果明确要求不缓存，
            # 也尝试停止模型
            if cache_model == "否":

                self.unload_model(
                    executable,
                    model
                )

            return (
                f"Ollama error:\n"
                f"{type(e).__name__}: {e}",
            )

    # =========================================================
    # Stop / unload model
    # =========================================================

    @staticmethod
    def unload_model(
            executable,
            model
    ):

        try:

            result = subprocess.run(
                [
                    executable,
                    "stop",
                    model,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )

            output = result.stdout or ""

            print(
                "[Cache] ollama stop:"
            )

            if output.strip():

                print(
                    output.strip()
                )

            if result.returncode == 0:

                print(
                    f"[Cache] Model stopped: {model}"
                )

            else:

                print(
                    f"[Cache] Failed to stop model "
                    f"(exit code {result.returncode})"
                )

        except Exception as e:

            print(
                "[Cache] Failed to unload model:",
                e
            )

    # =========================================================
    # Detect media paths
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

        # -----------------------------------------------------
        # Windows paths
        # -----------------------------------------------------

        windows_pattern = re.compile(
            r'(?:"([^"]+)"|\'([^\']+)\'|([A-Za-z]:[\\/][^\s"\']+))',
            re.MULTILINE
        )

        # -----------------------------------------------------
        # Linux paths
        # -----------------------------------------------------

        linux_pattern = re.compile(
            r'(?:"([^"]+)"|\'([^\']+)\'|(/[^ \t\r\n"\']+))',
            re.MULTILINE
        )

        candidates = []

        for match in windows_pattern.findall(prompt):

            candidates.extend(
                [
                    x
                    for x in match
                    if x
                ]
            )

        for match in linux_pattern.findall(prompt):

            candidates.extend(
                [
                    x
                    for x in match
                    if x
                ]
            )

        # -----------------------------------------------------
        # Validate paths
        # -----------------------------------------------------

        for path in candidates:

            path = path.strip()

            path = path.rstrip(
                ".,;:!?，。；：！？）)"
            )

            lower = path.lower()

            if not lower.endswith(
                    extensions
            ):
                continue

            if os.path.isfile(path):

                real_path = os.path.abspath(
                    path
                )

                if real_path not in paths:

                    paths.append(
                        real_path
                    )

        return paths


# =============================================================
# Node registration
# =============================================================

