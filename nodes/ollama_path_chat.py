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

        print("\n" + "=" * 70)
        print("Ollama Qwen3.8 Path Chat")
        print("=" * 70)

        print(f"Model       : {model}")
        print(f"Cache Model : {cache_model}")
        print(f"Ollama      : {executable}")

        command = [
            executable,
            "run",
            model,
            prompt,
        ]

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

            try:

                process.wait(
                    timeout=timeout
                )

            except subprocess.TimeoutExpired:

                print("[ERROR] Ollama timeout")

                process.kill()

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

            # -------------------------------------------------
            # 缓存控制
            # -------------------------------------------------

            if cache_model == "是":

                print(
                    f"[Cache] Keep model loaded: {model}"
                )

            else:

                print(
                    f"[Cache] Stopping model: {model}"
                )

                self.unload_model(
                    executable,
                    model
                )

            return (
                result,
            )

        except Exception as e:

            if cache_model == "否":

                self.unload_model(
                    executable,
                    model
                )

            return (
                f"Ollama error:\n"
                f"{type(e).__name__}: {e}",
            )

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

            if output.strip():
                print(
                    "[Ollama Stop]",
                    output.strip()
                )

            if result.returncode == 0:

                print(
                    f"[Ollama Stop] "
                    f"Model stopped: {model}"
                )

            else:

                print(
                    f"[Ollama Stop] Failed "
                    f"(exit code "
                    f"{result.returncode})"
                )

        except Exception as e:

            print(
                "[Ollama Stop] Error:",
                e
            )


class OllamaStopModel:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    "STRING",
                    {
                        "default": "qwen3.8:27b",
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
            },
        }

    RETURN_TYPES = ("STRING",)

    RETURN_NAMES = ("status",)

    FUNCTION = "stop"

    CATEGORY = "Ollama/Qwen3.8"

    OUTPUT_NODE = False

    def stop(
            self,
            model,
            ollama_path="/opt/ollama/ollama",
    ):

        executable = ollama_path.strip()

        if not executable:
            executable = "ollama"

        print("\n" + "=" * 70)
        print("Ollama Stop Model")
        print("=" * 70)

        print(f"Model    : {model}")
        print(f"Ollama   : {executable}")

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
                timeout=60,
            )

            output = (
                    result.stdout or ""
            ).strip()

            if result.returncode == 0:

                status = (
                    f"Successfully stopped "
                    f"model: {model}"
                )

                print(
                    f"[Ollama Stop] {status}"
                )

                if output:
                    print(output)

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
                    f"[Ollama Stop] {status}"
                )

                return (
                    status,
                )

        except FileNotFoundError:

            status = (
                f"Ollama executable not found: "
                f"{executable}"
            )

            print(
                f"[Ollama Stop] {status}"
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
                f"[Ollama Stop] {status}"
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
                f"[Ollama Stop] {status}"
            )

            return (
                status,
            )

