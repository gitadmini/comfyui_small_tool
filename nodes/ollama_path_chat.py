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
    # Ollama Chat
    # =========================================================

    def chat(
            self,
            prompt,
            model,
            cache_model="是",
            ollama_path="/opt/ollama/ollama",
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

        # =====================================================
        # Detect media paths
        # =====================================================

        media_paths = self.extract_media_paths(prompt)

        if media_paths:

            print("\nDetected media files:")

            for path in media_paths:

                print(
                    f"  {path}"
                )

        else:

            print(
                "\nNo local media file detected."
            )

        # =====================================================
        # Check media files
        # =====================================================

        for path in media_paths:

            if not os.path.isfile(path):

                print(
                    f"[WARNING] File does not exist: "
                    f"{path}"
                )

            else:

                size = os.path.getsize(path)

                print(
                    f"[MEDIA] {path} "
                    f"({size / 1024 / 1024:.2f} MB)"
                )

        # =====================================================
        # Disable terminal animation / color
        # =====================================================

        env = os.environ.copy()

        env["TERM"] = "dumb"

        env["NO_COLOR"] = "1"

        # 防止某些终端程序检测到 TTY 后启用动画
        env["CI"] = "1"

        # =====================================================
        # Ollama command
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

                env=env,
            )

            output_lines = []

            # =================================================
            # Read Ollama output
            # =================================================

            def reader():

                try:

                    for line in process.stdout:

                        clean_line = (
                            self.clean_ollama_output(
                                line
                            )
                        )

                        if clean_line:

                            print(
                                "[Ollama]",
                                clean_line
                            )

                            output_lines.append(
                                clean_line + "\n"
                            )

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
            # Wait for Ollama
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

                thread.join(
                    timeout=5
                )

                # ---------------------------------------------
                # cache_model = 否
                # 自动 stop
                # ---------------------------------------------

                if cache_model == "否":

                    self.unload_model(
                        executable,
                        model
                    )

                return (
                    f"Ollama timeout after "
                    f"{timeout} seconds.",
                )

            thread.join(
                timeout=10
            )

            raw_result = "".join(
                output_lines
            )

            result = self.clean_ollama_output(
                raw_result
            )

            # =================================================
            # Ollama failed
            # =================================================

            if process.returncode != 0:

                print(
                    f"[ERROR] Ollama exit code: "
                    f"{process.returncode}"
                )

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

                print(
                    "\n[Cache] Model cache enabled."
                )

                print(
                    f"[Cache] Keep model loaded: "
                    f"{model}"
                )

            else:

                print(
                    "\n[Cache] Model cache disabled."
                )

                print(
                    f"[Cache] Stopping model: "
                    f"{model}"
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

            print(
                "[Ollama Error]",
                e
            )

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
    # Clean Ollama output
    # =========================================================

    @staticmethod
    def clean_ollama_output(text):

        if not text:

            return ""

        # -----------------------------------------------------
        # ANSI escape sequences
        #
        # 例如：
        #
        # \x1b[?2026h
        # \x1b[?25l
        # \x1b[1G
        # \x1b[K
        # -----------------------------------------------------

        ansi_escape = re.compile(
            r"\x1B(?:"
            r"[@-_]"
            r"|\[[0-?]*[ -/]*[@-~]"
            r")"
        )

        text = ansi_escape.sub(
            "",
            text
        )

        # -----------------------------------------------------
        # Unicode ANSI / terminal control sequences
        # -----------------------------------------------------

        text = re.sub(
            r"\x1B\][^\x07]*(?:\x07|\x1B\\)",
            "",
            text
        )

        # -----------------------------------------------------
        # ASCII control characters
        # -----------------------------------------------------

        text = re.sub(
            r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]",
            "",
            text
        )

        # -----------------------------------------------------
        # Ollama spinner
        # -----------------------------------------------------

        text = re.sub(
            r"[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]",
            "",
            text
        )

        # -----------------------------------------------------
        # Remove carriage return
        # -----------------------------------------------------

        text = text.replace(
            "\r",
            ""
        )

        # -----------------------------------------------------
        # Remove "Added image '...'"
        # -----------------------------------------------------

        text = re.sub(
            r"Added image\s+'.*?'\s*",
            "",
            text,
            flags=re.MULTILINE
        )

        # -----------------------------------------------------
        # Remove "Added video '...'"
        # -----------------------------------------------------

        text = re.sub(
            r"Added video\s+'.*?'\s*",
            "",
            text,
            flags=re.MULTILINE
        )

        # -----------------------------------------------------
        # Remove repeated blank lines
        # -----------------------------------------------------

        text = re.sub(
            r"\n{3,}",
            "\n\n",
            text
        )

        return text.strip()

    # =========================================================
    # Stop Ollama Model
    # =========================================================

    @staticmethod
    def unload_model(
            executable,
            model,
    ):

        try:

            env = os.environ.copy()

            env["TERM"] = "dumb"

            env["NO_COLOR"] = "1"

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

                timeout=60,

                env=env,
            )

            output = (
                    result.stdout or ""
            ).strip()

            output = (
                OllamaPathChat.clean_ollama_output(
                    output
                )
            )

            if output:

                print(
                    "[Ollama Stop]",
                    output
                )

            if result.returncode == 0:

                print(
                    f"[Ollama Stop] "
                    f"Model stopped: "
                    f"{model}"
                )

            else:

                print(
                    f"[Ollama Stop] "
                    f"Failed to stop model "
                    f"(exit code "
                    f"{result.returncode})"
                )

        except Exception as e:

            print(
                "[Ollama Stop] Error:",
                e
            )

    # =========================================================
    # Extract image / video paths
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
        # Windows path
        # =====================================================

        windows_pattern = re.compile(
            r'(?:"([^"]+)"|'
            r"'([^']+)'|"
            r'([A-Za-z]:[\\/][^\s"\']+))',
            re.MULTILINE
        )

        # =====================================================
        # Linux path
        # =====================================================

        linux_pattern = re.compile(
            r'(?:"([^"]+)"|'
            r"'([^']+)'|"
            r'(/[^\s"\']+))',
            re.MULTILINE
        )

        candidates = []

        # -----------------------------------------------------
        # Windows
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Linux
        # -----------------------------------------------------

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

        # =====================================================
        # Validate
        # =====================================================

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
# Ollama Stop Node
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
            ollama_path="/opt/ollama/ollama",
    ):

        executable = (
            ollama_path.strip()
        )

        if not executable:

            executable = "ollama"

        print("\n")
        print("=" * 70)
        print("Ollama Stop Model")
        print("=" * 70)

        print(
            f"Model  : {model}"
        )

        print(
            f"Ollama : {executable}"
        )

        try:

            env = os.environ.copy()

            env["TERM"] = "dumb"

            env["NO_COLOR"] = "1"

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

                timeout=60,

                env=env,
            )

            output = (
                    result.stdout or ""
            )

            output = (
                OllamaPathChat.clean_ollama_output(
                    output
                )
            )

            if result.returncode == 0:

                status = (
                    f"Successfully stopped "
                    f"model: {model}"
                )

                print(
                    f"[Ollama Stop] "
                    f"{status}"
                )

                if output:

                    print(
                        output
                    )

                return (
                    status,
                )

            else:

                status = (
                    f"Failed to stop model "
                    f"{model}, "
                    f"exit code: "
                    f"{result.returncode}\n"
                    f"{output}"
                )

                print(
                    f"[Ollama Stop] "
                    f"{status}"
                )

                return (
                    status,
                )

        except FileNotFoundError:

            status = (
                "Ollama executable not found: "
                f"{executable}"
            )

            print(
                f"[Ollama Stop] "
                f"{status}"
            )

            return (
                status,
            )

        except subprocess.TimeoutExpired:

            status = (
                f"Timeout while stopping "
                f"model: {model}"
            )

            print(
                f"[Ollama Stop] "
                f"{status}"
            )

            return (
                status,
            )

        except Exception as e:

            status = (
                f"Error stopping model: "
                f"{type(e).__name__}: {e}"
            )

            print(
                f"[Ollama Stop] "
                f"{status}"
            )

            return (
                status,
            )

