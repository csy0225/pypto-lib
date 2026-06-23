# Dev Workflow Gotchas

A catalog of **non-pypto** time-sinks encountered while developing on
pypto-lib — Python tooling, environment activation, source control,
and host plumbing. Each entry has the same shape as
[known-pypto-pitfalls.md](known-pypto-pitfalls.md):

- **Symptom** — what you see
- **Trigger** — the smallest workflow step that reproduces it
- **Why** — the underlying mechanism (so you can recognise variants)
- **Workaround** — the recipe that gets you unblocked
- **Reproducer** — when applicable, the exact commands

Keep this list current. When you spend more than ~15 minutes debugging
something that turns out to be a known workflow gotcha (not a
pypto-compiler / kernel bug), add it here in the same shape.

For *pypto / pto-isa / simpler* hard limits (UB alignment, `pl.range`
unroll, `pl.dynamic` quirks, AICPU constraints) see
[known-pypto-pitfalls.md](known-pypto-pitfalls.md). For pypto-side
debugging (golden replay, runtime hangs, dump-tensor) see
[debugging.md](debugging.md).

---

## 1. Stale `__pycache__/*.pyc` after a test that monkey-patches module globals

**Symptom** — A fresh Python process imports `models.step3p5.config`
and reads back values that **contradict the source `.py` file**. A
downstream `pl` op fails with constants that should be impossible from
the source:

```
ValueError: tensor.set_validshape valid_cols (48) exceeds tensor bound 16
  at <kernel>.py:<line>
```

The check uses `NUM_HEADS_SWA_LOCAL` (source says `96 // 8 = 12`,
runtime reads `48`) against `NUM_HEADS_SWA_LOCAL_PAD` (source says
`16`, runtime reads `16`).

**Trigger** — Some earlier test in the same workspace ran one of the
ST/UT setup helpers that overwrite module-level constants at import
time, e.g.:

```python
# tests/step3p5/_perrank_setup.py  (or _tp1_setup.py)
def apply_perrank_patch(reload_modules=None):
    cfg = importlib.import_module("models.step3p5.config")
    cfg.TP_WORLD_SIZE = 1
    cfg.EP_WORLD_SIZE = 1
    ...
```

After the helper runs and Python's normal import finishes, the
interpreter writes a `__pycache__/config.cpython-311.pyc` whose
serialized module body reflects the **patched** values (in this case
`TP_WORLD_SIZE = 2` from a long-since-forgotten earlier experiment).

**Why** — Python's pyc invalidation only compares the **`.py` source
mtime** to the **mtime header in the pyc**. It does **not** compare
the post-`exec(code)` module dict against the source. Any monkey-
patched values that were live when the module was first imported get
captured into the pyc's marshalled code object as default values. The
next fresh process that imports the module gets those values back —
even though the `.py` on disk still says something else.

Two confirmations that this is what you are looking at:

```bash
# 1) Direct exec (no pyc) vs normal import disagree
python -c "exec(open('models/step3p5/config.py').read()); print(TP_WORLD_SIZE)"   # → 8
python -c "from models.step3p5 import config; print(config.TP_WORLD_SIZE)"        # → 2  ← stale pyc

# 2) Decode the pyc and dump the value
python -c '
import marshal
data = open("models/step3p5/__pycache__/config.cpython-311.pyc", "rb").read()
code = marshal.loads(data[16:])
ns = {}; exec(code, ns)
print("pyc TP_WORLD_SIZE =", ns["TP_WORLD_SIZE"])'
```

**Workaround** — bump the source `.py` mtime past the pyc so the
invalidation check fires:

```bash
find models/step3p5 -name "*.py" -exec touch {} +
```

`touch` is preferred over `rm -rf __pycache__/` because the latter is
flagged as destructive and the former achieves the same invalidation
without losing other pyc artefacts. Run this after **any** test that
calls `apply_perrank_patch` / `apply_tp1_patch` / `cfg.X = Y`, before
the next fresh `python -m ...` invocation.

**Better long-term fix** — have the setup helpers `setUp` /
`tearDown` symmetrically (restore the pre-patch values when the test
exits), or move the patches into a `with` context manager. Until that
lands, treat `touch` as the safety net.

**Reproducer** — surfaced 2026-06-22 on the smoke probe right after a
machine reboot. `_smoke_program_build` rc=1 with the `valid_cols (48)
exceeds bound 16` message; entire chain was a stale pyc from a
previous-session `apply_perrank_patch(TP=2)` experiment. Source code
on disk was clean.

---

## 2. Environment activation requires three separate `source`s

**Symptom** — first command in a fresh shell:

```
OSError: Environment variable 'ASCEND_HOME_PATH' is not set.
  at simpler/env_manager.py:27
```

Then, after fixing that:

```
OSError: PTO-ISA not available.
  Either export PTO_ISA_ROOT=/path/to/pto-isa, or manually clone to ...
```

**Trigger** — running `python <kernel>.py` from a fresh shell after
only sourcing the workspace `activate.sh`.

**Why** — the workspace `activate.sh` only activates the **Python
venv** and sets `PTOAS_ROOT` / `PYPTO_PROG_BUILD_DIR`. It does **not**
source the CANN `set_env.sh` (provides `ASCEND_HOME_PATH`,
`ASCEND_TOOLKIT_HOME`, `LD_LIBRARY_PATH` for CANN libs) and does **not**
export `PTO_ISA_ROOT` (the simpler `KernelCompiler` needs this to find
the pto-isa source tree it links against).

**Workaround** — make the three sources explicit at the top of every
session, in this order:

```bash
source /usr/local/Ascend/cann-9.0.0-beta.1/set_env.sh
source <workspace>/activate.sh
export PTO_ISA_ROOT=<workspace>/pto-isa
```

(For Phase 16 multi-card e2e the CANN path is **specifically**
`cann-9.0.0-beta.1`, not GA — see project CLAUDE.md "Phase 16 落地版本要
求" for why.)

**Better long-term fix** — extend `activate.sh` to detect the CANN
install and source it, and to export `PTO_ISA_ROOT` from a workspace-
relative path. Left as a follow-up; until then keep the three lines
in your shell history / a project-local `direnv` config.

**Reproducer** — observed every session on a netboot/tmpfs host
(driver / CANN paths survive in NVMe symlinks, environment variables
do not survive shell exit).

---

## 3. `git push` over HTTPS hangs with default HTTP/2

**Symptom** — `git push` to `github.com` over HTTPS hangs and times
out at ~130 s:

```
$ git push -u fork stepfun/develop:stepfun/develop
fatal: unable to access 'https://github.com/...': Operation timed out
```

`curl https://github.com` works fine from the same host.

**Trigger** — running `git push` against a network where HTTP/2 is
filtered or rate-limited at the egress, while curl negotiates HTTP/1.1
by default and works.

**Why** — git defaults to HTTP/2 for HTTPS remotes (`http.version` is
unset → libcurl picks H2). Some egress paths drop H2 frames silently
once the upload payload (pack data) starts streaming. The TCP
connection stays half-open until git's keepalive expires.

**Workaround** — force HTTP/1.1 for git operations:

```bash
git -c http.version=HTTP/1.1 push ...
git -c http.version=HTTP/1.1 fetch ...
```

For a persistent fix in a single repo:

```bash
git config http.version HTTP/1.1
```

**Reproducer** — observed on the dev host (netboot, behind corporate
egress); every `git push` / `git fetch` against HTTPS GitHub remotes
needed the `-c http.version=HTTP/1.1` prefix. SSH origin would have
worked, but see §4.

---

## 4. SSH-to-github auth on a netboot host loses keys on reboot

**Symptom** — `git push` against an SSH remote:

```
git@github.com: Permission denied (publickey).
fatal: Could not read from remote repository.
```

`ssh -T git@github.com` returns the same. `~/.ssh/authorized_keys`
exists; `~/.ssh/id_*` does **not**.

**Trigger** — fresh shell on a host whose `/` is netboot/tmpfs and
whose home directory rides on persistent NVMe but whose `~/.ssh/`
contents were never seeded (or were wiped by cluster provisioning).

**Why** — the typical "private key for outbound SSH" never gets
written on these hosts. Cluster provisioning lays down
`authorized_keys` for **incoming** SSH but not a private key for
**outgoing** GitHub operations.

**Workaround** — push over HTTPS using a short-lived Personal Access
Token via an **ephemeral URL** (token never lands in `.git/config`):

```bash
# Read token from a 600-perm file (don't echo to stdout):
PAT="$(tr -d '\n\r' < /path/to/secrets/github.env)"

git -c http.version=HTTP/1.1 push \
    "https://x-access-token:${PAT}@github.com/<org>/<repo>.git" \
    <local-branch>:<remote-branch>

unset PAT
```

If you must run this from a different host than the one holding the
token, scp the token file to a `mode 600` path in `/tmp` on the push
host, run the push, then `shred -u` (or `rm -f`) the temp file.

**Do not** add the URL with embedded token as a persistent
`git remote` — it will land in `.git/config` and leak via `git
remote -v`.

**Better long-term fix** — generate a workspace-side SSH keypair,
register the public key in GitHub (per-user fork or org-wide deploy
key), and persist `~/.ssh/id_ed25519` on the NVMe. Until that is done,
treat HTTPS+PAT as the canonical push path.

---

## 5. `gh` CLI is unavailable on host / pod

**Symptom** — `gh pr view`, `gh api`, `gh issue` all fail:

```
bash: gh: command not found
```

`apt-get install` is blocked or the pod image does not ship `gh`.

**Trigger** — needing to view a PR, list issues, or open one from a
host that has only `git`, `curl`, and Python.

**Workaround** — drive the GitHub REST API directly with `curl` and a
PAT. Examples:

```bash
PAT="$(tr -d '\n\r' < /path/to/secrets/github.env)"

# View PR
curl -sH "Authorization: token ${PAT}" \
     -H "Accept: application/vnd.github+json" \
     https://api.github.com/repos/<org>/<repo>/pulls/<num> | jq

# List open issues for a label
curl -sH "Authorization: token ${PAT}" \
     "https://api.github.com/repos/<org>/<repo>/issues?labels=bug&state=open" | jq

# Create issue (POST)
curl -sH "Authorization: token ${PAT}" \
     -H "Accept: application/vnd.github+json" \
     -X POST \
     -d '{"title":"...","body":"..."}' \
     https://api.github.com/repos/<org>/<repo>/issues
```

`jq` is usually available; if not, `python -m json.tool` is a
fallback.

**Reproducer** — `gh` unavailable on every netboot dev host and pod
encountered so far. Used the curl path to file simpler#1036 and
simpler#1037 in earlier sessions.

---

## 6. Cross-references

- [known-pypto-pitfalls.md](known-pypto-pitfalls.md) — pypto / pto-isa
  / simpler hard limits and bugs at the **kernel / codegen** layer.
  This file complements it with workflow / tooling gotchas.
- [debugging.md](debugging.md) — runtime / precision triage *after* a
  kernel compiles and runs; you generally hit those issues after
  you've gotten past everything in this file.
- [compile-runtime-workflow.md](compile-runtime-workflow.md) — the
  intended happy-path through `python <kernel>.py -p <platform>`.
- `../CLAUDE.md` (project-level) — phase-by-phase tracker; per-session
  milestones reference back to entries here when they burn debugging
  time.
