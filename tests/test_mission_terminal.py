"""Exercise real TTY editing; no model server or Gazebo required."""

import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time
import unittest


@unittest.skipUnless(sys.platform.startswith("linux"), "Linux PTY regression")
class TerminalTests(unittest.TestCase):
    def test_japanese_backspace_delete_and_cursor_edit_in_posix_locale(self):
        import pty

        root = Path(__file__).resolve().parent.parent
        code = """
import json,sys
sys.path.insert(0,'scripts')
from run_language_mission import configure_terminal_input
configure_terminal_input()
for i in range(3):
    text=input('INPUT> ')
    print('RESULT='+json.dumps(text,ensure_ascii=True),flush=True)
"""
        master, slave = pty.openpty()
        env = dict(
            os.environ,
            LANG="C",
            LC_ALL="C",
            LC_CTYPE="C",
            PYTHONUTF8="1",
            TERM="xterm-256color",
        )
        process = subprocess.Popen(
            [sys.executable, "-c", code],
            cwd=root,
            env=env,
            stdin=slave,
            stdout=slave,
            stderr=slave,
            close_fds=True,
        )
        os.close(slave)
        pending = b""

        def until(marker):
            nonlocal pending
            deadline = time.monotonic() + 10
            while marker not in pending:
                if time.monotonic() > deadline:
                    self.fail("TTY timed out: " + repr(pending))
                if select.select([master], [], [], 0.1)[0]:
                    try:
                        chunk = os.read(master, 65536)
                    except OSError:
                        self.fail("TTY closed: " + repr(pending))
                    if not chunk:
                        self.fail("TTY ended: " + repr(pending))
                    pending += chunk
            before, pending = pending.split(marker, 1)
            return before

        cases = [
            ("精密機器".encode() + b"\x7f" + "箱\n".encode(), "精密機箱"),
            ("工具箱".encode() + b"\x08" + "袋\n".encode(), "工具袋"),
            ("精密機器".encode() + b"\x1b[D\x7f" + "箱\n".encode(), "精密箱器"),
        ]
        try:
            for typed, expected in cases:
                until(b"INPUT> ")
                os.write(master, typed)
                until(b"RESULT=")
                line = until(b"\n").strip()
                self.assertEqual(json.loads(line), expected)
            self.assertEqual(process.wait(timeout=5), 0)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
            os.close(master)


if __name__ == "__main__":
    unittest.main()
