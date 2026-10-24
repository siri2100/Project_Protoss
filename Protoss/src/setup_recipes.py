"""Installation/data recipes formerly embedded as README shell blocks."""

from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

GROOT_REVISION = "51d4c89f72fda44cbf77285c6a8114b52676b8a1"
PUBLIC_GIT = ["git", "-c", "credential.helper=", "-c", "http.extraHeader=",
              "-c", "http.version=HTTP/1.1", "-c", "protocol.version=0"]
SYSTEM_PACKAGES = """git git-lfs curl build-essential ffmpeg vulkan-tools libvulkan1
libglvnd0 libgl1 libegl1 libglu1-mesa libxt6 libx11-6 libxext6 libxrender1
libxrandr2 libxinerama1 libxcursor1 libxi6 libxkbcommon0 libgomp1 libsm6 libice6""".split()


def setup_system(ctx, extra):
    if not ctx.args.dry_run and (platform.system(), platform.machine()) != ("Linux", "x86_64"):
        raise RuntimeError("setup-system requires Linux x86_64; run this on RunPod")
    ctx.run(["nvidia-smi", "-L"])
    ctx.run(["apt-get", "update"])
    ctx.run(["apt-get", "install", "-y", *SYSTEM_PACKAGES])
    ctx.run(["ldconfig"])
    if ctx.args.dry_run:
        ctx.run(["curl", "-LsSf", "https://astral.sh/uv/install.sh", "-o", "/tmp/protoss-uv-install.sh"])
        ctx.run(["sh", "/tmp/protoss-uv-install.sh"])
    else:
        with tempfile.TemporaryDirectory(prefix="protoss-uv-") as directory:
            installer = Path(directory) / "install.sh"
            ctx.run(["curl", "-LsSf", "https://astral.sh/uv/install.sh", "-o", installer])
            ctx.run(["sh", installer])
    ctx.run(["git", "lfs", "install"])


def update_project(ctx, extra):
    if not (ctx.root / ".git").exists():
        raise FileNotFoundError(f"Not a project checkout: {ctx.root}")
    ctx.run([*PUBLIC_GIT, "fetch", "origin", "v1.7:refs/remotes/origin/v1.7"])
    try:
        ctx.run(["git", "show-ref", "--verify", "--quiet", "refs/heads/v1.7"])
        command = ["git", "switch", "v1.7"]
    except subprocess.CalledProcessError:
        command = ["git", "switch", "-c", "v1.7", "--track", "origin/v1.7"]
    ctx.run(command)
    ctx.run(["git", "merge", "--ff-only", "origin/v1.7"])
    if not ctx.args.dry_run:
        for name in ("train.py", "prepare_droid.py", "src/prepare_droid.py"):
            if not (ctx.tools / name).is_file():
                raise FileNotFoundError(ctx.tools / name)


def folders(ctx, extra):
    for path in (ctx.data / "source", ctx.data / "trainset", ctx.checkpoints):
        ctx.mkdir(path)
        print(f"Storage: {path}")


def _public_git_env(ctx):
    env = dict(ctx.env)
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL="/dev/null")
    for key in ("GIT_CONFIG", "GIT_CONFIG_COUNT", "GIT_CONFIG_PARAMETERS", "GIT_ASKPASS",
                "SSH_ASKPASS", "GIT_TERMINAL_PROMPT"):
        env.pop(key, None)
    return env


def hf(ctx, arguments, env=None):
    ctx.run(["uv", "run", "--no-sync", "hf", *arguments], cwd=ctx.groot, env=env)


def verify_hf_login(ctx, env=None):
    # Older `hf auth whoami` exits 0 even when it prints "Not logged in".
    # The API raises on missing/invalid credentials, so check=True is reliable.
    ctx.run(["uv", "run", "--no-sync", "python", "-c",
             'from huggingface_hub import HfApi; print(HfApi().whoami()["name"])'],
            cwd=ctx.groot, env=env)


def download_base(ctx, env=None):
    ctx.mkdir(ctx.checkpoints)
    hf(ctx, ["download", "nvidia/Cosmos-Reason2-2B", "config.json", "--local-dir",
             str(ctx.tools / "access-check/cosmos")], env)
    hf(ctx, ["download", "nvidia/GR00T-N1.7-DROID", "--local-dir", str(ctx.base)], env)


def setup_groot(ctx, extra):
    env = _public_git_env(ctx)
    if ctx.groot.exists() and not (ctx.groot / ".git").is_dir():
        if ctx.args.dry_run:
            print(f"Would preserve incomplete checkout: {ctx.groot}")
        else:
            backup = Path(tempfile.mkdtemp(prefix=ctx.groot.name + ".incomplete.", dir=ctx.root))
            shutil.move(str(ctx.groot), backup / "source")
            print(f"Preserved incomplete checkout: {backup / 'source'}")
    if not (ctx.groot / ".git").is_dir():
        stage = (ctx.root / ".n17-clone.DRY-RUN" if ctx.args.dry_run else
                 Path(tempfile.mkdtemp(prefix=".n17-clone.", dir=ctx.root)))
        clone_env = {**env, "GIT_LFS_SKIP_SMUDGE": "1"}
        ctx.run([*PUBLIC_GIT, "clone", "--no-checkout", "https://github.com/NVIDIA/Isaac-GR00T.git",
                 stage / "source"], env=clone_env)
        ctx.run(["git", "lfs", "install", "--local"], cwd=stage / "source", env=env)
        ctx.run([*PUBLIC_GIT, "checkout", GROOT_REVISION], cwd=stage / "source", env=clone_env)
        if not ctx.args.dry_run:
            shutil.move(str(stage / "source"), ctx.groot)
            stage.rmdir()
    revision = ctx.run(["git", "rev-parse", "HEAD"], cwd=ctx.groot, env=env, capture=True)
    if not ctx.args.dry_run and revision != GROOT_REVISION:
        raise ValueError(f"Unexpected GR00T revision {revision}; expected {GROOT_REVISION}. Preserve the old checkout first.")
    ctx.run([*PUBLIC_GIT, "submodule", "update", "--init", "--recursive"], cwd=ctx.groot, env=env)
    ctx.run([*PUBLIC_GIT, "lfs", "pull", "--include=scripts/deployment/dgpu/wheels/**"], cwd=ctx.groot, env=env)
    ctx.run(["uv", "sync", "--locked", "--python", "3.12"], cwd=ctx.groot)
    try:
        verify_hf_login(ctx)
    except subprocess.CalledProcessError:
        hf(ctx, ["auth", "login"])
        verify_hf_login(ctx)
    download_base(ctx)


def login(ctx, extra):
    # Explicit recovery action: stale exported tokens must not override the new login.
    env = dict(ctx.env)
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HUB_DISABLE_IMPLICIT_TOKEN"):
        env.pop(key, None)
    hf(ctx, ["auth", "login"], env)
    verify_hf_login(ctx, env)
    download_base(ctx, env)


def setup_data(ctx, extra):
    ctx.run(["uv", "pip", "install", "--python", ctx.python,
             "huggingface-hub>=0.34,<2", "pyarrow>=18,<24", "pandas>=2.2,<3", "scipy>=1.13,<2"], cwd=ctx.groot)
    ctx.python_code('import torch, numpy, pandas, pyarrow, scipy, huggingface_hub; '
                    'print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda); '
                    'print(torch.ones(1, device="cuda"), torch.cuda.get_device_name(0))')
    ctx.groot_run("train.py", ["--help"])


def download_droid(ctx, extra):
    ctx.groot_run("prepare_droid.py", ["download", "--dataset-dir", ctx.data / "source",
                                      "--num-episodes", str(ctx.config.DOWNLOAD_EPISODES), *extra])


def prepare_data(ctx, extra):
    ctx.groot_run("prepare_groot_dataset.py", ["--groot-root", ctx.groot,
        "--dataset-dir", ctx.data / "source", "--output-dir", ctx.data,
        "--style-method", "original", "--seed", "42", *extra])


def setup_robolab(ctx, extra):
    ctx.run(["bash", ctx.tools / "setup_robolab_595.sh", "setup", *extra])


def smoke_robolab(ctx, extra):
    ctx.run(["bash", ctx.tools / "setup_robolab_595.sh", "smoke", *extra])


def setup_robolab51(ctx, extra):
    root = Path(ctx.config.ROBOLAB_51_ROOT)
    env = {**ctx.env, "OMNI_KIT_ACCEPT_EULA": "Y", "UV_PROJECT_ENVIRONMENT": ".venv-51"}
    if not root.exists():
        ctx.run(["git", "clone", "https://github.com/NVlabs/RoboLab.git", root])
    revision = ctx.run(["git", "rev-parse", "HEAD"], cwd=root, capture=True)
    ctx.write(root / "protoss_v12_robolab_revision.txt", revision + "\n")
    ctx.run(["uv", "sync", "--python", "3.11", "--extra", "isaac51"], cwd=root, env=env)
    ctx.run([root / ".venv-51/bin/python", "-c",
             'import isaaclab; from isaaclab.app import AppLauncher; print("Isaac Lab import OK")'], cwd=root, env=env)
    patch_client51(ctx, extra)


def patch_client51(ctx, extra):
    target = Path(ctx.config.ROBOLAB_51_ROOT) / "policies/gr00t/client.py"
    if ctx.args.dry_run:
        print(f"Would back up once and copy {ctx.tools / 'src/robolab_gr00t_client.py'} to {target}")
    else:
        if not target.is_file():
            raise FileNotFoundError(target)
        backup = target.with_name(target.name + ".before_protoss_history")
        if not backup.exists():
            shutil.copyfile(target, backup)
        shutil.copyfile(ctx.tools / "src/robolab_gr00t_client.py", target)
    root = Path(ctx.config.ROBOLAB_51_ROOT)
    patch = ctx.tools / "patches/robolab_fresh_results.patch"
    try:
        ctx.run(["git", "apply", "--reverse", "--check", patch], cwd=root)
    except subprocess.CalledProcessError:
        ctx.run(["git", "apply", "--check", patch], cwd=root)
        ctx.run(["git", "apply", patch], cwd=root)
    if not ctx.args.dry_run:
        shutil.copyfile(ctx.tools / "src/robolab_fresh_results.py", root / "robolab/protoss_fresh_results.py")
