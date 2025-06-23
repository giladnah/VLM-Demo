#!/usr/bin/env python3
"""
VLM App Runner script to launch both the FastAPI backend (app.py) and the Gradio UI (gradio_runner.py).

- Starts both services as subprocesses.
- Streams their logs to the console with clear prefixes.
- If either process exits, the script will terminate both.

Usage:
    python vlm_app_runner.py

Requirements:
    - Python 3.7+
    - Both app.py and gradio_runner.py must be present in the same directory or importable.
"""
import subprocess
import threading
import sys
import os
import signal

def stream_logs(proc, prefix):
    for line in iter(proc.stdout.readline, b''):
        sys.stdout.write(f"[{prefix}] {line.decode().rstrip()}\n")
        sys.stdout.flush()

def main():
    processes = {}
    try:
        print("[VLMAppRunner] Starting FastAPI app (unbuffered)...")
        app_proc = subprocess.Popen(
            [sys.executable, "-u", "app.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1
        )
        print(f"[VLMAppRunner] FastAPI app started with PID {app_proc.pid}")
        processes['app'] = app_proc
        app_thread = threading.Thread(target=stream_logs, args=(app_proc, "FastAPI"), daemon=True)
        app_thread.start()

        print("[VLMAppRunner] Starting Gradio runner (unbuffered)...")
        gradio_proc = subprocess.Popen(
            [sys.executable, "-u", "gradio_runner.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1
        )
        print(f"[VLMAppRunner] Gradio runner started with PID {gradio_proc.pid}")
        processes['gradio'] = gradio_proc
        gradio_thread = threading.Thread(target=stream_logs, args=(gradio_proc, "Gradio"), daemon=True)
        gradio_thread.start()

        print("[VLMAppRunner] Both FastAPI and Gradio processes started. Press Ctrl+C to stop.")

        # Wait for either process to exit
        while True:
            for name, proc in processes.items():
                ret = proc.poll()
                if ret is not None:
                    print(f"[VLMAppRunner] {name} process (PID {proc.pid}) exited with code {ret}. Shutting down both services.")
                    for p in processes.values():
                        if p.poll() is None:
                            p.terminate()
                    return
            threading.Event().wait(0.5)
    except KeyboardInterrupt:
        print("[VLMAppRunner] KeyboardInterrupt received. Terminating both services...")
        for proc in processes.values():
            if proc.poll() is None:
                proc.terminate()
    finally:
        for proc in processes.values():
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
        print("[VLMAppRunner] All processes terminated.")

if __name__ == "__main__":
    main()
