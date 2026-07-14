#!/usr/bin/env python3
"""Build gsplat with channel-template instantiations split across CUDA TUs.

The stock 3DGUT serial and parallel forward sources instantiate every supported
channel count in one translation unit. Ninja can only schedule at source-file
granularity, so those files become long single-process build tails. This script
stages temporary wrapper .cu files, each compiling a disjoint channel subset,
and a small CPU-side dispatcher that preserves the existing public ABI.

The original CUDA sources and kernel math are not edited. They are temporarily
renamed to .inc files so setuptools' ``csrc/*.cu`` glob does not compile the
monolithic translation units, while each generated wrapper includes the same
implementation with group-specific preprocessor names.
"""

from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
CSRC = REPO_ROOT / "gsplat" / "cuda" / "csrc"
CONFIG = CSRC / "Config.h"
HEADER = CSRC / "RasterizeToPixelsFromWorld3DGS.h"

SERIAL_SOURCE = CSRC / "RasterizeToPixelsFromWorld3DGSSerialBatchFwd.cu"
PARALLEL_SOURCE = CSRC / "RasterizeToPixelsFromWorld3DGSParallelBatchFwd.cu"
SERIAL_STAGED = CSRC / "RasterizeToPixelsFromWorld3DGSSerialBatchFwd.channel_split.inc"
PARALLEL_STAGED = CSRC / "RasterizeToPixelsFromWorld3DGSParallelBatchFwd.channel_split.inc"
DISPATCHER = CSRC / "RasterizeToPixelsFromWorld3DGSChannelSplit.cpp"
LOCK = CSRC / ".channel_split_build.lock"

SERIAL_PUBLIC = "launch_rasterize_to_pixels_from_world_3dgs_serial_batch_fwd_kernel"
SERIAL_IMPL = "launch_rasterize_to_pixels_from_world_3dgs_serial_batch_fwd_impl"
PARALLEL_PUBLIC = "launch_rasterize_to_pixels_from_world_3dgs_parallel_batch_fwd_kernel"

GENERATED_PREFIXES = (
    "RasterizeToPixelsFromWorld3DGSSerialBatchFwdChannels",
    "RasterizeToPixelsFromWorld3DGSParallelBatchFwdChannels",
)


def _parse_channel_list(value: str) -> list[int]:
    channels: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            channel = int(item)
        except ValueError as exc:
            raise ValueError(f"invalid channel count {item!r}") from exc
        if channel <= 0:
            raise ValueError(f"channel counts must be positive, got {channel}")
        channels.append(channel)
    if not channels:
        raise ValueError("channel list is empty")
    if len(channels) != len(set(channels)):
        raise ValueError(f"channel list contains duplicates: {channels}")
    return sorted(channels)


def _default_channels() -> list[int]:
    env_value = os.environ.get("NUM_CHANNELS")
    if env_value:
        return _parse_channel_list(env_value)

    text = CONFIG.read_text(encoding="utf-8")
    match = re.search(
        r"^\s*#\s*define\s+GSPLAT_NUM_CHANNELS\s+([^\n]+)$",
        text,
        flags=re.MULTILINE,
    )
    if not match:
        raise RuntimeError(f"could not find GSPLAT_NUM_CHANNELS in {CONFIG}")
    return _parse_channel_list(match.group(1))


def _channel_weight(channel: int) -> float:
    # High-CDIM kernels carry CDIM-sized local arrays and unrolled channel loops.
    # A super-linear estimate gives 256/512/513 their own jobs before packing
    # the much cheaper low-channel specializations together.
    return float(channel) ** 1.35 + 32.0


def _partition_channels(channels: Sequence[int], group_count: int) -> list[list[int]]:
    group_count = max(1, min(group_count, len(channels)))
    groups: list[list[int]] = [[] for _ in range(group_count)]
    loads = [0.0] * group_count

    for channel in sorted(channels, key=_channel_weight, reverse=True):
        index = min(range(group_count), key=lambda i: loads[i])
        groups[index].append(channel)
        loads[index] += _channel_weight(channel)

    groups = [sorted(group) for group in groups if group]
    groups.sort(key=lambda group: min(group))

    flattened = [channel for group in groups for channel in group]
    if sorted(flattened) != sorted(channels):
        raise AssertionError("partition lost or duplicated a channel")
    return groups


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", text)


def _extract_parameter_text(header: str, function_name: str) -> str:
    match = re.search(rf"\bvoid\s+{re.escape(function_name)}\s*\(", header)
    if not match:
        raise RuntimeError(f"could not find declaration for {function_name}")

    open_paren = header.find("(", match.start())
    depth = 0
    for index in range(open_paren, len(header)):
        char = header[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return header[open_paren + 1 : index]
    raise RuntimeError(f"unterminated declaration for {function_name}")


def _split_parameters(parameter_text: str) -> list[str]:
    clean = _strip_comments(parameter_text)
    parts: list[str] = []
    start = 0
    angle = paren = bracket = brace = 0

    for index, char in enumerate(clean):
        if char == "<":
            angle += 1
        elif char == ">":
            angle = max(0, angle - 1)
        elif char == "(":
            paren += 1
        elif char == ")":
            paren -= 1
        elif char == "[":
            bracket += 1
        elif char == "]":
            bracket -= 1
        elif char == "{":
            brace += 1
        elif char == "}":
            brace -= 1
        elif char == "," and angle == paren == bracket == brace == 0:
            part = clean[start:index].strip()
            if part:
                parts.append(part)
            start = index + 1

    tail = clean[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _parameter_names(parameter_text: str) -> list[str]:
    names: list[str] = []
    for parameter in _split_parameters(parameter_text):
        parameter = re.sub(r"\s*=.*$", "", parameter, flags=re.DOTALL).strip()
        match = re.search(r"([A-Za-z_]\w*)\s*(?:\[[^\]]*\])?\s*$", parameter)
        if not match:
            raise RuntimeError(f"could not extract parameter name from: {parameter!r}")
        names.append(match.group(1))
    return names


def _group_symbol(base: str, index: int) -> str:
    return f"{base}_channel_group_{index:02d}"


def _wrapper_text(kind: str, index: int, channels: Sequence[int]) -> str:
    channel_csv = ", ".join(str(channel) for channel in channels)
    if kind == "serial":
        source = SERIAL_STAGED.name
        public_group = _group_symbol(SERIAL_PUBLIC, index)
        impl_group = _group_symbol(SERIAL_IMPL, index)
        alias = f"GsplatSerialSupportedChannelsGroup{index:02d}"
        renames = f"""\
#define SupportedChannels {alias}
#define {SERIAL_IMPL} {impl_group}
#define {SERIAL_PUBLIC} {public_group}
"""
        undefs = f"""\
#undef {SERIAL_PUBLIC}
#undef {SERIAL_IMPL}
#undef SupportedChannels
"""
    elif kind == "parallel":
        source = PARALLEL_STAGED.name
        public_group = _group_symbol(PARALLEL_PUBLIC, index)
        alias = f"GsplatParallelSupportedChannelsGroup{index:02d}"
        renames = f"""\
#define SupportedChannels {alias}
#define {PARALLEL_PUBLIC} {public_group}
"""
        undefs = f"""\
#undef {PARALLEL_PUBLIC}
#undef SupportedChannels
"""
    else:
        raise ValueError(f"unknown wrapper kind: {kind}")

    return f"""// Generated by scripts/build_split_cuda.py. Do not edit.\n\
#ifdef GSPLAT_NUM_CHANNELS\n\
#undef GSPLAT_NUM_CHANNELS\n\
#endif\n\
#define GSPLAT_NUM_CHANNELS {channel_csv}\n\
{renames}\
#include \"{source}\"\n\
{undefs}\
#undef GSPLAT_NUM_CHANNELS\n"""


def _dispatcher_text(groups: Sequence[Sequence[int]]) -> str:
    header = HEADER.read_text(encoding="utf-8")
    serial_params = _extract_parameter_text(header, SERIAL_PUBLIC)
    parallel_params = _extract_parameter_text(header, PARALLEL_PUBLIC)
    serial_args = ", ".join(_parameter_names(serial_params))
    parallel_args = ", ".join(_parameter_names(parallel_params))

    def declarations(name: str, params: str) -> str:
        return "\n".join(
            f"void {_group_symbol(name, index)}({params});"
            for index in range(len(groups))
        )

    def cases(name: str, args: str) -> str:
        lines: list[str] = []
        for index, group in enumerate(groups):
            labels = "\n".join(f"        case {channel}:" for channel in group)
            lines.append(
                f"{labels}\n"
                f"            return {_group_symbol(name, index)}({args});"
            )
        return "\n".join(lines)

    supported = ", ".join(str(channel) for group in groups for channel in group)
    return f"""// Generated by scripts/build_split_cuda.py. Do not edit.\n\
#include \"Config.h\"\n\
\n\
#if GSPLAT_BUILD_3DGUT\n\
\n\
#include <ATen/core/Tensor.h>\n\
#include <c10/util/Exception.h>\n\
\n\
#include \"RasterizeToPixelsFromWorld3DGS.h\"\n\
\n\
namespace gsplat\n\
{{\n\
{declarations(SERIAL_PUBLIC, serial_params)}\n\
\n\
{declarations(PARALLEL_PUBLIC, parallel_params)}\n\
\n\
void {SERIAL_PUBLIC}({serial_params})\n\
{{\n\
    const int32_t channels = static_cast<int32_t>(colors.size(-1));\n\
    switch(channels)\n\
    {{\n\
{cases(SERIAL_PUBLIC, serial_args)}\n\
        default:\n\
            TORCH_CHECK_VALUE(false, \"Unsupported number of channels: \", channels, \"; compiled values are {{{supported}}}\");\n\
    }}\n\
}}\n\
\n\
void {PARALLEL_PUBLIC}({parallel_params})\n\
{{\n\
    const int32_t channels = static_cast<int32_t>(colors.size(-1));\n\
    switch(channels)\n\
    {{\n\
{cases(PARALLEL_PUBLIC, parallel_args)}\n\
        default:\n\
            TORCH_CHECK_VALUE(false, \"Unsupported number of channels: \", channels, \"; compiled values are {{{supported}}}\");\n\
    }}\n\
}}\n\
}} // namespace gsplat\n\
\n\
#endif\n"""


def _generated_paths() -> Iterable[Path]:
    yield DISPATCHER
    for path in CSRC.glob("*.cu"):
        if path.name.startswith(GENERATED_PREFIXES):
            yield path


def _recover_stale_state() -> None:
    pairs = ((SERIAL_SOURCE, SERIAL_STAGED), (PARALLEL_SOURCE, PARALLEL_STAGED))
    for original, staged in pairs:
        if not original.exists() and staged.exists():
            staged.replace(original)
        elif original.exists() and staged.exists():
            raise RuntimeError(
                f"both {original.name} and {staged.name} exist; refusing to guess which one is authoritative"
            )
    for path in list(_generated_paths()):
        path.unlink(missing_ok=True)
    LOCK.unlink(missing_ok=True)


@contextlib.contextmanager
def _staged_sources(groups: Sequence[Sequence[int]], keep: bool):
    if LOCK.exists():
        raise RuntimeError(
            f"{LOCK} exists; another split build may be active. "
            "Run this script with --recover after confirming no build is running."
        )
    if not SERIAL_SOURCE.exists() or not PARALLEL_SOURCE.exists():
        raise RuntimeError("expected monolithic 3DGUT forward CUDA sources are missing")

    LOCK.write_text(str(os.getpid()), encoding="utf-8")
    generated: list[Path] = []
    try:
        SERIAL_SOURCE.replace(SERIAL_STAGED)
        PARALLEL_SOURCE.replace(PARALLEL_STAGED)

        for index, group in enumerate(groups):
            serial_wrapper = CSRC / f"{GENERATED_PREFIXES[0]}{index:02d}.cu"
            parallel_wrapper = CSRC / f"{GENERATED_PREFIXES[1]}{index:02d}.cu"
            serial_wrapper.write_text(_wrapper_text("serial", index, group), encoding="utf-8")
            parallel_wrapper.write_text(_wrapper_text("parallel", index, group), encoding="utf-8")
            generated.extend((serial_wrapper, parallel_wrapper))

        DISPATCHER.write_text(_dispatcher_text(groups), encoding="utf-8")
        generated.append(DISPATCHER)
        yield generated
    finally:
        if keep:
            print("Split sources were left staged by request.", file=sys.stderr)
            print(f"Run `{sys.executable} {Path(__file__)} --recover` to restore the tree.", file=sys.stderr)
        else:
            for path in generated:
                path.unlink(missing_ok=True)
            if SERIAL_STAGED.exists():
                SERIAL_STAGED.replace(SERIAL_SOURCE)
            if PARALLEL_STAGED.exists():
                PARALLEL_STAGED.replace(PARALLEL_SOURCE)
            LOCK.unlink(missing_ok=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--groups",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="number of channel compilation groups (default: min(8, CPU count))",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="stage generated sources without running pip; implies --keep-staged",
    )
    parser.add_argument(
        "--keep-staged",
        action="store_true",
        help="do not restore generated files after the build (for debugging)",
    )
    parser.add_argument(
        "--recover",
        action="store_true",
        help="restore originals and remove files from an interrupted split build",
    )
    parser.add_argument(
        "pip_args",
        nargs=argparse.REMAINDER,
        help="arguments after -- are appended to `python -m pip install`",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.recover:
        _recover_stale_state()
        print("Recovered the source tree from split-build staging.")
        return 0

    if args.groups <= 0:
        raise ValueError("--groups must be positive")

    channels = _default_channels()
    groups = _partition_channels(channels, args.groups)
    print("CUDA channel compilation groups:")
    for index, group in enumerate(groups):
        print(f"  {index:02d}: {', '.join(map(str, group))}")

    keep = args.keep_staged or args.prepare_only
    with _staged_sources(groups, keep=keep) as generated:
        print(f"Staged {len(generated)} generated compilation units.")
        if args.prepare_only:
            return 0

        pip_args = list(args.pip_args)
        if pip_args and pip_args[0] == "--":
            pip_args.pop(0)
        if not pip_args:
            pip_args = ["--no-build-isolation", "-v", "."]
        elif "--no-build-isolation" not in pip_args:
            pip_args.insert(0, "--no-build-isolation")

        env = os.environ.copy()
        env.setdefault("MAX_JOBS", str(max(1, min(len(groups), os.cpu_count() or 1))))
        command = [sys.executable, "-m", "pip", "install", *pip_args]
        print("Running:", " ".join(command))
        subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
