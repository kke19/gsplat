from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_split_cuda.py"
SPEC = importlib.util.spec_from_file_location("build_split_cuda", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_default_channel_partition_is_complete_and_disjoint():
    channels = MODULE._default_channels()
    groups = MODULE._partition_channels(channels, 8)
    flattened = [channel for group in groups for channel in group]

    assert sorted(flattened) == sorted(channels)
    assert len(flattened) == len(set(flattened))
    assert len(groups) == 8


def test_high_channel_specializations_are_not_forced_into_one_group():
    channels = MODULE._default_channels()
    groups = MODULE._partition_channels(channels, 8)
    group_for = {
        channel: index
        for index, group in enumerate(groups)
        for channel in group
    }

    assert group_for[512] != group_for[513]
    assert group_for[256] != group_for[257]


def test_wrappers_rename_public_symbols_and_limit_channels():
    serial = MODULE._wrapper_text("serial", 3, [32, 33])
    parallel = MODULE._wrapper_text("parallel", 3, [32, 33])

    assert "#define GSPLAT_NUM_CHANNELS 32, 33" in serial
    assert MODULE._group_symbol(MODULE.SERIAL_PUBLIC, 3) in serial
    assert MODULE._group_symbol(MODULE.SERIAL_IMPL, 3) in serial
    assert MODULE.SERIAL_STAGED.name in serial

    assert "#define GSPLAT_NUM_CHANNELS 32, 33" in parallel
    assert MODULE._group_symbol(MODULE.PARALLEL_PUBLIC, 3) in parallel
    assert MODULE.PARALLEL_STAGED.name in parallel


def test_dispatcher_covers_every_compiled_channel():
    channels = MODULE._default_channels()
    groups = MODULE._partition_channels(channels, 8)
    dispatcher = MODULE._dispatcher_text(groups)

    for channel in channels:
        assert f"case {channel}:" in dispatcher
    for index in range(len(groups)):
        assert MODULE._group_symbol(MODULE.SERIAL_PUBLIC, index) in dispatcher
        assert MODULE._group_symbol(MODULE.PARALLEL_PUBLIC, index) in dispatcher
