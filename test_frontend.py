"""Exercise the Streamlit form against a real API and simulated API failures."""

import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import requests
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


class FrontendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            cls.port = listener.getsockname()[1]
        cls.api_url = f"http://127.0.0.1:{cls.port}"
        cls.log = tempfile.TemporaryFile(mode="w+")
        cls.server = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "api:app", "--host", "127.0.0.1",
             "--port", str(cls.port)], cwd=ROOT, stdout=cls.log, stderr=subprocess.STDOUT,
        )
        cls.addClassCleanup(cls.cleanup_server)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if cls.server.poll() is not None:
                break
            try:
                if requests.get(f"{cls.api_url}/health", timeout=1).status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(0.2)
        cls.log.seek(0)
        raise RuntimeError("Test API did not start:\n" + cls.log.read())

    @classmethod
    def cleanup_server(cls):
        if cls.server.poll() is None:
            cls.server.terminate()
            try:
                cls.server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                cls.server.kill()
                cls.server.wait()
        cls.log.close()

    def new_app(self):
        with patch.dict(os.environ, {"LOAN_API_URL": self.api_url}):
            app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=20).run()
        self.assertFalse(app.exception)
        return app

    def test_form_calls_real_api(self):
        with patch.dict(os.environ, {"LOAN_API_URL": self.api_url}):
            app = self.new_app()
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertIn("Approved", app.success[0].value)
            self.assertEqual(app.metric[0].label, "Model approval score")
            # Change just the score and exercise the opposite prediction.
            app.number_input[0].set_value(420)
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertIn("Rejected", app.warning[0].value)

    def test_request_failures_are_shown_in_form(self):
        for error, message in [
            (requests.ConnectionError("offline"), "Cannot reach"),
            (requests.Timeout("timeout"), "timed out"),
        ]:
            with self.subTest(error=error):
                app = self.new_app()
                with patch("requests.post", side_effect=error):
                    app.button[0].click().run()
                self.assertFalse(app.exception)
                self.assertIn(message, app.error[0].value)

    def test_server_and_malformed_responses_are_shown_in_form(self):
        cases = [
            (503, {}, "unavailable"),
            (422, {"detail": [{"loc": ["body", "cibil_score"], "msg": "Invalid score"}]}, "check your inputs"),
            (200, {"approval_score": "not-a-number"}, "invalid response"),
            (200, [], "invalid response"),
            (500, {}, "HTTP 500"),
        ]
        for status, body, message in cases:
            with self.subTest(status=status, body=body):
                response = Mock(status_code=status)
                response.json.return_value = body
                if status >= 400:
                    response.raise_for_status.side_effect = requests.HTTPError(response=response)
                app = self.new_app()
                with patch("requests.post", return_value=response):
                    app.button[0].click().run()
                self.assertFalse(app.exception)
                self.assertIn(message, app.error[0].value)


if __name__ == "__main__":
    unittest.main()
