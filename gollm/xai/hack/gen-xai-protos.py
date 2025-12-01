#!/usr/bin/env -S uv run -s -U
# /// script
# requires-python = ">=3.14"
# dependencies = [
# "grpcio",
# "protobuf",
# "pydantic",
# "requests",
# "aiohttp",
# "packaging",
# "opentelemetry-sdk",
# ]
# ///

"""
Regenerate Go protobufs for proto files that are not published in xai-proto
but are embedded as descriptors inside xai-sdk-python.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from google.protobuf import descriptor_pb2


GO_MODULE = "github.com/zchee/tumix/gollm/xai"
DEFAULT_REF = "main"
PROTO_FILES = [
    "xai/api/v1/collections.proto",
    "xai/api/v1/shared.proto",
    "xai/api/v1/types.proto",
]
MAPPINGS = {
    "xai/api/v1/collections.proto": f"{GO_MODULE}/api/v1/collectionspb",
    "xai/api/v1/shared.proto": f"{GO_MODULE}/api/v1/sharedpb",
    "xai/api/v1/types.proto": f"{GO_MODULE}/api/v1/ragpb",
}
VT_FEATURES = "size+equal+marshal+marshal_strict+unmarshal+unmarshal_unsafe+clone+pool"


@dataclass(frozen=True)
class Paths:
    root: Path
    tmp: Path
    clone: Path
    desc: Path


def require(cmd: str) -> None:
    if not shutil.which(cmd):
        sys.exit(f"missing dependency: {cmd}")


def clone_python_sdk(paths: Paths, ref: str) -> None:
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            ref,
            "https://github.com/xai-org/xai-sdk-python.git",
            str(paths.clone),
        ],
        check=True,
    )


def build_descriptor_set(paths: Paths) -> None:
    sys.path.insert(0, str(paths.clone / "src"))

    modules = [
        "xai_sdk.proto.v6.collections_pb2",
        "xai_sdk.proto.v6.types_pb2",
        "xai_sdk.proto.v6.shared_pb2",
    ]

    seen: set[str] = set()
    fdset = descriptor_pb2.FileDescriptorSet()

    def add(fd) -> None:
        if fd.name in seen:
            return
        seen.add(fd.name)
        fdset.file.append(descriptor_pb2.FileDescriptorProto.FromString(fd.serialized_pb))
        for dep in fd.dependencies:
            add(dep)

    for mod_name in modules:
        mod = importlib.import_module(mod_name)
        add(mod.DESCRIPTOR)

    paths.desc.parent.mkdir(parents=True, exist_ok=True)
    paths.desc.write_bytes(fdset.SerializeToString())


def protoc_generate(paths: Paths) -> None:
    go_opt = "module=" + GO_MODULE + "," + ",".join(
        f"M{src}={dst}" for src, dst in MAPPINGS.items()
    )
    grpc_opt = "module=" + GO_MODULE + ",require_unimplemented_servers=true," + ",".join(
        f"M{src}={dst}" for src, dst in MAPPINGS.items()
    )
    vt_opt = "module=" + GO_MODULE + ",features=" + VT_FEATURES + "," + ",".join(
        f"M{src}={dst}" for src, dst in MAPPINGS.items()
    )

    cmd: list[str] = [
        "protoc",
        f"--descriptor_set_in={paths.desc}",
        *PROTO_FILES,
        f"--go_out={paths.root}",
        f"--go_opt={go_opt}",
        f"--go-grpc_out={paths.root}",
        f"--go-grpc_opt={grpc_opt}",
        f"--go-vtproto_out={paths.root}",
        f"--go-vtproto_opt={vt_opt}",
    ]

    subprocess.run(cmd, check=True)


def main() -> None:
    ref = os.environ.get("XAI_SDK_PYTHON_REF", DEFAULT_REF)
    tmp_base = Path(os.environ.get("XAI_PY_PROTO_TMP", tempfile.gettempdir())) / "xai-sdk-python-proto"

    paths = Paths(
        root=Path(__file__).resolve().parents[1],
        tmp=tmp_base,
        clone=tmp_base / "src",
        desc=tmp_base / "xai-sdk-python.desc",
    )

    if paths.tmp.exists():
        shutil.rmtree(paths.tmp)
    paths.tmp.mkdir(parents=True, exist_ok=True)

    clone_python_sdk(paths, ref)
    build_descriptor_set(paths)
    protoc_generate(paths)
    print(f"Regenerated collections/shared/types protos from xai-sdk-python @ {ref}")


if __name__ == "__main__":
    import shutil

    tool_dir = Path(__file__).parent.parent.parent.parent / "tools" / "bin"
    if not tool_dir.is_dir():
        sys.exit(f"missing tools directory: {tool_dir}")
    os.environ["PATH"] += os.pathsep + str(tool_dir)

    require("git")
    require("protoc")
    require("protoc-gen-go")
    require("protoc-gen-go-grpc")
    require("protoc-gen-go-vtproto")
    main()
