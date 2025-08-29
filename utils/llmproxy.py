from google import genai
import time
from google.genai.errors import ClientError, ServerError
import os
from dotenv import load_dotenv


class LLMProxy:
    def __init__(self, requests_per_minute=10):
        load_dotenv()
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY")).models
        self.requests_per_minute = requests_per_minute
        self.interval = 60 / requests_per_minute
        self.last_request_time = 0
        self.models = [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
            "gemini-2.5-flash-lite",
        ]  # Tried in Order
        self.max_retries = 6

    def wait_if_needed(self):
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.interval:
            time.sleep(self.interval - elapsed)
        self.last_request_time = time.time()

    def __call__(self, contents):
        self.wait_if_needed()

        for index, model_name in enumerate(self.models):
            for attempt in range(1, self.max_retries + 1):
                try:
                    print(
                        f"[INFO] Attempting model {model_name} (attempt {attempt}/{self.max_retries + 1})..."
                    )
                    resp = self.client.generate_content(
                        contents=contents,
                        model=model_name,
                    )

                    if index != 0:
                        self.models.insert(0, self.models.pop(index))
                    return resp.text.strip() if resp and resp.text else ""

                except ServerError or ClientError as e:
                    # This handles all transient, retriable errors.
                    if attempt <= self.max_retries:
                        print(
                            f"[WARN] Call to {model_name} failed (attempt {attempt}/{self.max_retries}): {e}. Retrying in {2**attempt} seconds..."
                        )
                        time.sleep(2**attempt)
                    else:
                        print(
                            f"[ERROR] All retries failed for {model_name}. Removing {model_name} and trying others."
                        )
                        break

                except Exception as e:
                    print(
                        f"[ERROR] Unexpected error with model {model_name}: {e}. Skipping to next model..."
                    )
                    break

        print("[FATAL] All model calls failed. Returning empty string.")
        return ""
