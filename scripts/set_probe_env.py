#!/usr/bin/env python3
"""Rewrite .xctestrun inside a Firebase XCTest zip with OnlyTestIdentifiers and env vars."""

import argparse
import os
import plistlib
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path


def collect_test_target_dicts(obj, out_list) -> None:
    if isinstance(obj, dict):
        if "TestBundlePath" in obj and "TestHostPath" in obj:
            out_list.append(obj)
        for key, val in obj.items():
            if key != "__xctestrun_metadata__":
                collect_test_target_dicts(val, out_list)
    elif isinstance(obj, list):
        for item in obj:
            collect_test_target_dicts(item, out_list)


def parse_env_pairs(raw_items: list[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for idx, raw in enumerate(raw_items, start=1):
        key, sep, value = raw.partition("=")
        key = key.strip()
        if sep != "=" or not key:
            raise SystemExit(
                f"Error: --env argument #{idx} must have format KEY=VALUE with a non-empty KEY"
            )
        if key[0].isdigit() or not all(ch.isalnum() or ch == "_" for ch in key):
            raise SystemExit(f"Error: --env argument #{idx} has invalid variable name {key!r}")
        pairs.append((key, value))
    return pairs


def clone_zip_info(src_zi: zipfile.ZipInfo) -> zipfile.ZipInfo:
    dst_zi = zipfile.ZipInfo(filename=src_zi.filename, date_time=src_zi.date_time)
    dst_zi.compress_type = src_zi.compress_type
    dst_zi.comment = src_zi.comment
    dst_zi.extra = src_zi.extra
    dst_zi.create_system = src_zi.create_system
    dst_zi.create_version = src_zi.create_version
    dst_zi.extract_version = src_zi.extract_version
    dst_zi.reserved = src_zi.reserved
    dst_zi.flag_bits = src_zi.flag_bits & ~0x08
    dst_zi.volume = src_zi.volume
    dst_zi.internal_attr = src_zi.internal_attr
    dst_zi.external_attr = src_zi.external_attr
    return dst_zi


def rewrite_xctest_zip(
    input_zip: Path,
    output_zip: Path,
    only_identifiers: list[str],
    env_pairs: list[tuple[str, str]],
) -> None:
    if not input_zip.is_file():
        raise SystemExit(f"Error: input zip does not exist: {input_zip}")

    cleaned_only = [item.strip() for item in only_identifiers if item.strip()]
    if len(cleaned_only) != len(only_identifiers):
        raise SystemExit("Error: --only identifier must not be empty")

    with zipfile.ZipFile(input_zip, "r") as src_zf:
        bad_entry = src_zf.testzip()
        if bad_entry is not None:
            raise SystemExit(f"Error: input zip failed CRC check on entry: {bad_entry}")

        root_xctestruns = [
            zi.filename
            for zi in src_zf.infolist()
            if not zi.filename.endswith("/")
            and "/" not in zi.filename
            and zi.filename.endswith(".xctestrun")
        ]
        if len(root_xctestruns) != 1:
            raise SystemExit(
                f"Error: expected exactly 1 .xctestrun at zip root, found: {root_xctestruns}"
            )
        xctestrun_name = root_xctestruns[0]

        plist_data = plistlib.loads(src_zf.read(xctestrun_name))
        targets: list[dict] = []
        collect_test_target_dicts(plist_data, targets)
        if not targets:
            raise SystemExit(f"Error: no test targets found inside {xctestrun_name}")

        for target in targets:
            if cleaned_only:
                target["OnlyTestIdentifiers"] = list(cleaned_only)
            if env_pairs:
                env_vars = dict(target.get("EnvironmentVariables") or {})
                test_env_vars = dict(target.get("TestingEnvironmentVariables") or {})
                for key, value in env_pairs:
                    env_vars[key] = value
                    test_env_vars[key] = value
                target["EnvironmentVariables"] = env_vars
                target["TestingEnvironmentVariables"] = test_env_vars

        rewritten_plist = plistlib.dumps(plist_data, fmt=plistlib.FMT_XML, sort_keys=True)

        output_zip.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=".set_probe_env_",
            suffix=".zip",
            dir=str(output_zip.parent),
        )
        os.close(fd)
        tmp_path = Path(tmp_name)

        try:
            with zipfile.ZipFile(
                tmp_path,
                "w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=6,
            ) as dst_zf:
                for src_zi in src_zf.infolist():
                    dst_zi = clone_zip_info(src_zi)
                    if src_zi.filename == xctestrun_name:
                        with dst_zf.open(dst_zi, "w") as wfh:
                            wfh.write(rewritten_plist)
                    elif src_zi.filename.endswith("/"):
                        dst_zf.writestr(dst_zi, b"")
                    else:
                        with src_zf.open(src_zi, "r") as rfh, dst_zf.open(dst_zi, "w") as wfh:
                            shutil.copyfileobj(rfh, wfh, 1024 * 1024)

            os.replace(tmp_path, output_zip)
        except BaseException:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    env_keys = [k for k, _ in env_pairs]
    print(
        f"[set_probe_env] Rewrote {xctestrun_name} -> {output_zip} "
        f"(targets={len(targets)}, only={cleaned_only}, env_keys={env_keys})"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite .xctestrun inside a built Firebase XCTest zip with "
            "OnlyTestIdentifiers and environment variables without printing secret values."
        )
    )
    parser.add_argument("input_zip", type=Path, help="Path to the built XCTest zip")
    parser.add_argument("output_zip", type=Path, help="Path for the rewritten XCTest zip")
    parser.add_argument(
        "--only",
        dest="only_identifiers",
        action="append",
        default=[],
        metavar="TEST_IDENTIFIER",
        help="Test identifier (<TestClass/testMethod>) for OnlyTestIdentifiers (repeatable)",
    )
    parser.add_argument(
        "--env",
        dest="env_items",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Environment variable pair added to EnvironmentVariables and TestingEnvironmentVariables (repeatable)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    if hasattr(parser, "parse_intermixed_args"):
        args = parser.parse_intermixed_args(argv)
    else:
        args = parser.parse_args(argv)

    env_pairs = parse_env_pairs(args.env_items)
    rewrite_xctest_zip(
        input_zip=args.input_zip,
        output_zip=args.output_zip,
        only_identifiers=args.only_identifiers,
        env_pairs=env_pairs,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
