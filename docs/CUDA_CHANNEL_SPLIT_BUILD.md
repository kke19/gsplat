# Parallel CUDA channel-instantiation build

The 3DGUT serial and parallel forward rasterizers instantiate every supported
feature-channel count in a single `.cu` translation unit. Ninja can parallelize
source files, but it cannot schedule the template specializations hidden inside
one NVCC invocation. On machines with many CPU cores, the build therefore ends
with one or two long NVCC tasks while most cores are idle.

`scripts/build_split_cuda.py` keeps the existing kernels and public C++ ABI, but
temporarily compiles disjoint channel groups as separate CUDA translation
units. The generated group wrappers rename the internal launcher symbols and
set a group-specific `GSPLAT_NUM_CHANNELS`; a generated CPU dispatcher forwards
the runtime channel count to the matching group. The original CUDA source text
is included unchanged in every group.

## Build

From the repository root:

```bash
python scripts/build_split_cuda.py --groups 8
```

The default command executed by the script is:

```bash
python -m pip install --no-build-isolation -v .
```

Additional pip arguments can be passed after `--`:

```bash
python scripts/build_split_cuda.py --groups 8 -- -e . -v
```

Existing build environment variables are preserved. For example:

```bash
MAX_JOBS=8 \
NVCC_FLAGS="--split-compile=4" \
python scripts/build_split_cuda.py --groups 8
```

Avoid setting both `MAX_JOBS` and `--split-compile` excessively high because
that creates nested parallelism and can exhaust RAM. A reasonable first test on
a high-core workstation is `MAX_JOBS=8`, eight channel groups, and either no
`--split-compile` or a small value such as 2–4.

`NUM_CHANNELS` remains supported. The script reads it and partitions only those
channels:

```bash
NUM_CHANNELS="3,16,32" python scripts/build_split_cuda.py --groups 3
```

## Inspect generated sources

To stage the wrappers without compiling:

```bash
python scripts/build_split_cuda.py --groups 8 --prepare-only
```

This intentionally leaves the two original forward `.cu` files renamed to
`.channel_split.inc`, creates the group wrappers and dispatcher under
`gsplat/cuda/csrc`, and writes a lock file. Restore the source tree afterwards:

```bash
python scripts/build_split_cuda.py --recover
```

The normal build path restores the source tree automatically, including after
a regular compiler failure or `Ctrl+C`. `--recover` is provided for hard process
termination or machine interruption.

## What changes during compilation

Before staging, Ninja sees two heavy translation units:

```text
RasterizeToPixelsFromWorld3DGSSerialBatchFwd.cu
RasterizeToPixelsFromWorld3DGSParallelBatchFwd.cu
```

During a split build it instead sees, for example:

```text
RasterizeToPixelsFromWorld3DGSSerialBatchFwdChannels00.cu
...
RasterizeToPixelsFromWorld3DGSSerialBatchFwdChannels07.cu
RasterizeToPixelsFromWorld3DGSParallelBatchFwdChannels00.cu
...
RasterizeToPixelsFromWorld3DGSParallelBatchFwdChannels07.cu
RasterizeToPixelsFromWorld3DGSChannelSplit.cpp
```

Each group contains complete serial or complete parallel-forward behavior for
its channel subset. In particular, the parallel partials, batch-scan and
batch-replay kernels remain together in each group, so their `CDIM`-dependent
CSR state layout cannot diverge.

## Validation requested after the first build

After compilation, run the existing 3DGUT forward/backward tests, including:

- tile sizes 8 and 16;
- color-only and returned-normal modes;
- hit-distance on and off;
- safe and unsafe masked-tile output;
- parallel forward-only and backward-compatible paths;
- representative low and high channel counts.

The source transformation is intended to preserve generated kernel behavior;
the main unknown until CUDA compilation is the toolchain/linker handling of the
separate host launch stubs and template specializations.
