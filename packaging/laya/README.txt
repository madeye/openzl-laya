OpenZL-Laya for macOS 14+ on Apple Silicon
==========================================

bin/zli                 OpenZL CLI with local Laya integer routing (--laya)
bin/openzl-laya-worker  Core ML model worker, started by zli on demand

Keep both files in the same directory on your PATH. Then download the model
once (about 940 MB, SHA-256 verified):

    openzl-laya-worker prepare

Compress an integer column with routing:

    zli compress input.bin --profile le-u64 --laya -o output.zl
    zli decompress output.zl -o input.copy

Without the model, --laya falls back to measuring candidates locally.
Decompression never needs the model or the worker.

These binaries are not notarized. If the archive was downloaded with a
browser, remove the quarantine attribute before running them:

    xattr -dr com.apple.quarantine <unpacked directory>

Documentation: https://github.com/madeye/openzl-laya/blob/dev/doc/laya.md
