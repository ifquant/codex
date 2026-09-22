"""Check package replacement and cache cleanup without compiling Rust."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class LocalBuildTest(unittest.TestCase):
    def test_failures_preserve_package_and_success_only_cleans_caches(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            script = root / "scripts/build-local-codex.sh"
            shutil.copyfile(Path(__file__).with_name(script.name), script)
            target = root / "codex-rs/target"
            output = target / "local-package"
            output.mkdir(parents=True)
            (output / "old").touch()
            cache = target / "dev-small/deps"
            cache.mkdir(parents=True)
            running_binary = target / "dev-small/codex"
            running_binary.touch()
            commands = root / "commands"
            commands.mkdir()
            mocks = {
                "uname": "echo Darwin",
                "cargo": 'exit "${BUILD_EXIT:-0}"',
                "just": """
exit_code=${PACKAGE_EXIT:-0}
if [ "$exit_code" != 0 ]; then exit "$exit_code"; fi
while [ "$1" != --package-dir ]; do shift; done
mkdir -p "$2/bin"
printf '#!/bin/sh\nexit "${VERSION_EXIT:-0}"\n' > "$2/bin/codex"
chmod +x "$2/bin/codex"
""",
            }
            for name, body in mocks.items():
                path = commands / name
                path.write_text("#!/bin/sh\n" + body + "\n")
                path.chmod(0o755)
            env = {
                **os.environ,
                "PATH": str(commands) + os.pathsep + os.environ["PATH"],
                "LIBCLANG_PATH": str(root),
            }
            for failure in [
                {"BUILD_EXIT": "1"},
                {"PACKAGE_EXIT": "1"},
                {"VERSION_EXIT": "1"},
                {},
            ]:
                result = subprocess.run(
                    ["bash", str(script), "--clean-cache"],
                    env={**env, **failure},
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, int(bool(failure)), result.stderr)
                self.assertEqual((output / "old").exists(), bool(failure))
                self.assertEqual(cache.exists(), bool(failure))
                self.assertTrue(running_binary.exists())
                self.assertFalse(list(target.glob(".local-package*")))
            self.assertTrue((output / "bin/codex").exists())


if __name__ == "__main__":
    unittest.main()
