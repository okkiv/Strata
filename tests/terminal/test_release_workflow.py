"""The release workflow (.github/workflows/release.yml): a maintainer starts it, it builds only Linux x86_64 AMD (HIP)
engines plus the linux-x86_64 Terminal, signs with the feed key from the repository secret, and publishes only on request.
These are the rules packaging/UPDATES.md states for a release; like test_docker_files.py this reads the YAML as text,
because only GitHub runs a workflow."""
from __future__ import annotations

import re
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"

#: a release made here builds no asset for these platforms; the words may only appear in a comment, an input description or a log line
FOREIGN = ("windows-latest", "macos", "cuda13", "cuda12", "sycl", "aarch64")


def build_lines(text: str):
    """The lines that say what is built: no comment, no description, no log line."""
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#") or "description:" in s or "echo" in s:
            continue
        yield line

ENV_SECRET = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*: \$\{\{ secrets\.[A-Za-z0-9_]+ \}\}$")


def triggers(data: dict) -> dict:
    return data.get("on", data.get(True))              # YAML 1.1 reads a bare `on` as True


def steps(data: dict, job: str) -> list[dict]:
    return data["jobs"][job]["steps"]


def commands(text: str) -> list[str]:
    """The shell commands with their `\\` continuations joined, so a flag on a wrapped line is read with its command."""
    out: list[str] = []
    pending = ""
    for line in text.splitlines():
        if pending:
            line = pending + " " + line.strip()
        if line.endswith("\\"):
            pending = line[:-1]
            continue
        pending = ""
        out.append(line)
    return out


def job_runs(data: dict, name: str) -> str:
    return "\n".join(step.get("run", "") for step in steps(data, name))


class Workflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = WORKFLOW.read_text(encoding="utf-8")
        cls.data = yaml.safe_load(cls.text)

    def test_the_workflow_exists_and_starts_by_hand_only(self):
        self.assertTrue(WORKFLOW.is_file())
        self.assertEqual(set(triggers(self.data)), {"workflow_dispatch"})
        inputs = triggers(self.data)["workflow_dispatch"]["inputs"]
        self.assertFalse(inputs["publish"]["default"])             # a run publishes nothing unless it is asked to
        self.assertFalse(inputs["prerelease"]["default"])
        for name in ("tag", "variants", "sequence", "docker"):
            self.assertIn(name, inputs)
        self.assertEqual(inputs["variants"]["default"], "rx9070,rx7900")

    def test_no_writes_in_the_top_level_permissions(self):
        self.assertEqual(self.data["permissions"], {"contents": "read"})
        self.assertEqual(self.data["jobs"]["publish"]["permissions"], {"contents": "write"})
        self.assertEqual(self.data["jobs"]["docker"]["permissions"], {"contents": "read", "packages": "write"})
        for name, job in self.data["jobs"].items():     # the two jobs that publish are the only ones escalated
            self.assertTrue(name in ("publish", "docker") or job.get("permissions") is None, name)

    def test_only_linux_x86_64_amd_is_built(self):
        for foreign in FOREIGN:
            for line in build_lines(self.text):
                self.assertNotIn(foreign, line, line)
        for name, job in self.data["jobs"].items():
            self.assertTrue(str(job["runs-on"]).startswith("ubuntu-"), name)
        for command in commands(job_runs(self.data, "terminal")):
            if "--platform" in command:
                self.assertIn("--platform linux-x86_64", command, command)
        self.assertEqual(self.data["jobs"]["engine-hip"]["container"]["image"], "ubuntu:22.04")   # glibc 2.35

    def test_the_feed_key_is_a_secret_and_never_sits_in_a_command(self):
        for name, job in self.data["jobs"].items():
            for step in job["steps"]:
                self.assertNotIn("${{ secrets.", step.get("run", ""), f"{name}/{step.get('name')}")
        for job in ("engine-hip", "feed"):     # the key is set on the step that signs, not on every step of the job
            self.assertNotIn("FEED_KEY_JSON", self.data["jobs"][job].get("env", {}), job)
        for line in self.text.splitlines():
            if "${{ secrets." in line:
                stripped = line.strip()
                if stripped.startswith("if:") or stripped.startswith("#"):
                    continue
                self.assertRegex(stripped, ENV_SECRET, stripped)
        feed = job_runs(self.data, "feed")
        engine = job_runs(self.data, "engine-hip")
        for script in (feed, engine):
            self.assertIn('printf \'%s\' "$FEED_KEY_JSON" > "$RUNNER_TEMP/feed-key.json"', script)
            self.assertIn('--key "$RUNNER_TEMP/feed-key.json"', script)
            self.assertIn("the secret FEED_KEY_JSON is empty", script)     # a missing secret is named, not a signing crash
        self.assertIn('--pub "$RUNNER_TEMP/feed-key.pub.json"', feed)
        self.assertIn("strata_control/update_keys.json", feed)          # the public half comes from the repo's trust store

    def test_the_package_version_comes_from_the_checkout(self):
        self.assertIsNone(re.search(r"PKGVER=[0-9]", self.text), "the package version comes from the checkout, never written here")
        self.assertIn('PKGVER="$PKGVER"', job_runs(self.data, "engine-hip"))
        self.assertIn("project(strata VERSION", job_runs(self.data, "plan"))    # engine version out of CMakeLists.txt
        self.assertIn('apps/desktop/src-tauri/Cargo.toml', job_runs(self.data, "plan"))
        self.assertIn('PKGVER: ${{ needs.plan.outputs.engine_version }}', self.text)

    def test_the_rocm_version_is_picked_by_setup_py_not_hardcoded(self):
        self.assertIsNone(re.search(r"\d+\.\d+\.\d+a\d{6,}", self.text), "a TheRock version must not be pinned in the YAML")
        run = job_runs(self.data, "engine-hip")
        self.assertIn("rocm_wanted", run)
        self.assertIn('"rocm[libraries,devel]==$VER"', run)
        self.assertIn("::warning::$NOTE", run)                        # a pruned index is a warning in the log, not a silent pin
        self.assertIn("--index-url", run)
        self.assertIn("--no-cache-dir", run)
        self.assertIn('path --root', run)                                   # rocm-sdk path --root is the SDK root
        self.assertIn("_rocm_sdk_libraries_", run)                          # the card family's libraries go to LIBW

    def test_the_archives_are_flat_the_installers_are_uploaded_and_the_sums_are_made(self):
        for job, tool in (("engine-hip", "pack-engine"), ("terminal", "pack-control")):
            hits = [c for c in commands(job_runs(self.data, job)) if tool in c]
            self.assertTrue(hits, tool)
            for command in hits:
                self.assertIn("--url-prefix=", command, command)
        feed = job_runs(self.data, "feed")
        self.assertIn('-type f -exec mv -t "$REL"', feed)          # every file of both folders, no glob that can miss
        self.assertNotIn('mv "$REL"/engine/*.tar.gz', feed)
        self.assertIn("packaging/install.sh", feed)
        self.assertIn("packaging/install.ps1", feed)
        self.assertIn("sha256sum", feed)
        self.assertIn("SHA256SUMS.txt", feed)

    def test_the_sequence_grows_by_one(self):
        plan = job_runs(self.data, "plan")
        self.assertIn("releases/download/$PREV/strata-feed.json", plan)     # by tag: a pre-release is a published feed too
        self.assertIn("select(.draft | not)", plan)                        # a draft was never live, so it sets no sequence
        self.assertIn('["sequence"]', plan)
        self.assertIn("+ 1 ))", plan)
        self.assertIn('--sequence "$SEQUENCE"', job_runs(self.data, "feed"))
        self.assertIsNone(re.search(r"--sequence\s+\d", self.text), "the sequence is derived, never written here")

    def test_the_container_gets_git_before_the_checkout_runs(self):
        """actions/checkout runs inside the job's container, and the ubuntu:22.04 image ships no git."""
        steps = self.data["jobs"]["engine-hip"]["steps"]
        names = [str(s.get("name", s.get("uses", ""))) for s in steps]
        git_step = next(i for i, n in enumerate(names) if "Build tools" in n)
        checkout = next(i for i, s in enumerate(steps) if "checkout" in str(s.get("uses", "")))
        self.assertLess(git_step, checkout, names)
        self.assertIn("git", steps[git_step]["run"])
        self.assertIn("-v /:/host", self.data["jobs"]["engine-hip"]["container"]["options"])
        self.assertIn("/host/usr/share/dotnet", job_runs(self.data, "engine-hip"))   # the runner's disk, not the image's

    def test_the_carry_job_keeps_the_release_complete(self):
        carry = job_runs(self.data, "carry")
        self.assertIn("gh release download \"$PREVIOUS\" --repo \"$GITHUB_REPOSITORY\" -p '*'", carry)
        self.assertIn("--dir \"$RUNNER_TEMP/carry\" || true", carry)      # a release with no assets is not an error
        self.assertIn("$(basename \"$f\")", carry)                        # the names this release writes are not carried
        self.assertIn("strata-engine-*-linux-x86_64-hip-*", carry)        # and the versions this release replaces are dropped
        self.assertIn("strata-control-*-linux-x86_64.tar.gz*", carry)
        self.assertIn("select(.draft | not)", carry)                      # carry from the newest published release
        self.assertIn("::error::no published release to carry from", carry)   # without one the release would be incomplete

    def test_the_release_points_at_the_commit_its_assets_come_from(self):
        publish = job_runs(self.data, "publish")
        self.assertIn('--target "$COMMIT"', publish)
        self.assertEqual(self.data["jobs"]["publish"]["env"]["COMMIT"], "${{ needs.plan.outputs.commit }}")

    def test_the_jobs_and_their_order(self):
        jobs = self.data["jobs"]
        self.assertEqual(list(jobs), ["plan", "engine-hip", "terminal", "feed", "carry", "publish", "docker"])
        self.assertEqual(jobs["engine-hip"]["needs"], "plan")
        self.assertEqual(set(jobs["feed"]["needs"]), {"plan", "engine-hip", "terminal"})
        self.assertEqual(jobs["carry"]["needs"], "feed")
        self.assertEqual(set(jobs["publish"]["needs"]), {"plan", "feed", "carry"})
        self.assertEqual(set(jobs["docker"]["needs"]), {"plan", "publish"})
        self.assertIn("inputs.publish", jobs["docker"]["if"])
        self.assertIn("inputs.docker", jobs["docker"]["if"])
        self.assertEqual(jobs["engine-hip"]["strategy"]["matrix"], {"include": "${{ fromJSON(needs.plan.outputs.matrix) }}"})
        self.assertEqual(jobs["engine-hip"]["strategy"]["fail-fast"], False)

    def test_publishing_is_gated_and_a_pre_release_is_announced(self):
        publish = job_runs(self.data, "publish")
        env = self.data["jobs"]["publish"]["env"]
        self.assertEqual(env["DRAFT"], "${{ !inputs.publish && '--draft' || '' }}")   # a run with publish off makes a draft
        self.assertEqual(env["PRE"], "${{ inputs.prerelease && '--prerelease' || '' }}")
        self.assertIn("$DRAFT", publish)
        self.assertIn("$PRE", publish)
        self.assertIn("gh release create", publish)
        self.assertIn('ghcr.io/okkiv/strata', self.text)
        self.assertIn("docker/publish.sh", self.text)
        warning = [s for s in steps(self.data, "publish") if "pre-release" in str(s.get("name", ""))]
        self.assertTrue(warning)
        self.assertIn("latest/download", warning[0]["run"])
        self.assertIn("install.sh", warning[0]["run"])
        self.assertIn("inputs.prerelease", str(warning[0].get("if")))
        self.assertEqual(self.data["concurrency"]["cancel-in-progress"], False)
        self.assertIn("inputs.tag", self.data["concurrency"]["group"])


if __name__ == "__main__":
    unittest.main()
