#!/usr/bin/env python3
"""Normalize permissions, package, and verify Firebase Test Lab iOS XCTest zip."""

import os
import plistlib
import shutil
import stat
import struct
import subprocess
import sys
import time
import zipfile
from pathlib import Path

MACHO_MAGICS = {
    b"\xfe\xed\xfa\xce",  # MH_MAGIC (32-bit BE)
    b"\xce\xfa\xed\xfe",  # MH_CIGAM (32-bit LE)
    b"\xfe\xed\xfa\xcf",  # MH_MAGIC_64 (64-bit BE)
    b"\xcf\xfa\xed\xfe",  # MH_CIGAM_64 (64-bit LE, arm64)
    b"\xca\xfe\xba\xbf",  # FAT_MAGIC_64
    b"\xbf\xba\xfe\xca",  # FAT_CIGAM_64
}

FAT_32_MAGICS = {
    b"\xca\xfe\xba\xbe",  # FAT_MAGIC
    b"\xbe\xba\xfe\xca",  # FAT_CIGAM
}


def is_executable_bytes(header: bytes) -> bool:
    if len(header) >= 2 and header[:2] == b"#!":
        return True
    if len(header) < 4:
        return False
    magic = header[:4]
    if magic in MACHO_MAGICS:
        return True
    if magic in FAT_32_MAGICS and len(header) >= 8:
        narch = (
            struct.unpack(">I", header[4:8])[0]
            if magic == b"\xca\xfe\xba\xbe"
            else struct.unpack("<I", header[4:8])[0]
        )
        return 1 <= narch <= 32
    return False


def is_executable_file(path: Path) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    try:
        with path.open("rb") as fh:
            header = fh.read(8)
        return is_executable_bytes(header)
    except OSError:
        return False


def collect_test_target_dicts(obj, out_list):
    if isinstance(obj, dict):
        if "TestBundlePath" in obj and "TestHostPath" in obj:
            out_list.append(obj)
        for key, val in obj.items():
            if key != "__xctestrun_metadata__":
                collect_test_target_dicts(val, out_list)
    elif isinstance(obj, list):
        for item in obj:
            collect_test_target_dicts(item, out_list)


def normalize_dir(staging_dir: Path) -> None:
    if not staging_dir.is_dir():
        raise SystemExit(f"Staging directory does not exist: {staging_dir}")

    dir_count = 0
    exec_files = []
    regular_count = 0

    os.chmod(staging_dir, 0o755)
    for root, dirs, files in os.walk(staging_dir):
        root_path = Path(root)
        dirs.sort()
        files.sort()
        for d in dirs:
            dp = root_path / d
            if not dp.is_symlink():
                os.chmod(dp, 0o755)
                dir_count += 1
        for f in files:
            fp = root_path / f
            if fp.is_symlink():
                continue
            if is_executable_file(fp):
                os.chmod(fp, 0o755)
                exec_files.append(fp.relative_to(staging_dir).as_posix())
            else:
                os.chmod(fp, 0o644)
                regular_count += 1

    print(
        f"[package_xctest_zip] Normalized {staging_dir}: "
        f"{dir_count} dirs (0755), {len(exec_files)} executables (0755), {regular_count} files (0644)"
    )
    for rel in exec_files:
        print(f"  [0755] {rel}")


def clean_staging(staging_dir: Path) -> None:
    if not staging_dir.is_dir():
        raise SystemExit(f"Staging directory does not exist: {staging_dir}")

    rel_dir = staging_dir / "Release-iphoneos"
    for extra in ("XCFrameworkIntermediates", "include"):
        target = rel_dir / extra
        if target.exists():
            shutil.rmtree(target)
            print(f"[package_xctest_zip] Removed {target}")

    xctestruns = sorted(staging_dir.glob("*.xctestrun"))
    top_llama = rel_dir / "llama.framework"
    if len(xctestruns) == 1 and top_llama.exists():
        raw_plist_text = plistlib.dumps(plistlib.loads(xctestruns[0].read_bytes())).decode("utf-8", "ignore")
        if "Release-iphoneos/llama.framework" not in raw_plist_text:
            shutil.rmtree(top_llama)
            print("[package_xctest_zip] Removed unreferenced top-level Release-iphoneos/llama.framework")

    to_remove_dirs = []
    to_remove_files = []
    for root, dirs, files in os.walk(staging_dir):
        rp = Path(root)
        for d in list(dirs):
            if d.endswith(".dSYM") or d.endswith(".swiftmodule"):
                to_remove_dirs.append(rp / d)
                dirs.remove(d)
        for f in files:
            if f == ".DS_Store":
                to_remove_files.append(rp / f)

    for dp in to_remove_dirs:
        shutil.rmtree(dp)
        print(f"[package_xctest_zip] Removed bundle intermediate: {dp.relative_to(staging_dir).as_posix()}")
    for fp in to_remove_files:
        fp.unlink()


def sign_dir(staging_dir: Path) -> None:
    if not staging_dir.is_dir():
        raise SystemExit(f"Staging directory does not exist: {staging_dir}")

    sign_targets = []
    bundle_suffixes = (".framework", ".xctest", ".app")
    for root, dirs, files in os.walk(staging_dir):
        rp = Path(root)
        for d in dirs:
            if d.endswith(bundle_suffixes):
                sign_targets.append(rp / d)
        for f in files:
            if f.endswith(".dylib"):
                sign_targets.append(rp / f)

    def sort_key(p: Path):
        rel_parts = p.relative_to(staging_dir).parts
        # Sign deepest paths first; at equal depth sign .dylib -> .framework -> .xctest -> .app
        suffix_rank = 3
        if p.name.endswith(".dylib"):
            suffix_rank = 0
        elif p.name.endswith(".framework"):
            suffix_rank = 1
        elif p.name.endswith(".xctest"):
            suffix_rank = 2
        return (-len(rel_parts), suffix_rank, p.as_posix())

    sign_targets.sort(key=sort_key)
    print(f"[package_xctest_zip] Ad-hoc signing {len(sign_targets)} items inside-out:")
    for target in sign_targets:
        rel = target.relative_to(staging_dir).as_posix()
        print(f"  codesign --force --sign - --timestamp=none {rel}", flush=True)
        subprocess.run(
            ["codesign", "--force", "--sign", "-", "--timestamp=none", str(target)],
            check=True,
        )


def zip_datetime_from_mtime(mtime: float):
    lt = time.localtime(mtime)
    year = max(1980, min(2107, lt.tm_year))
    return (year, lt.tm_mon, lt.tm_mday, lt.tm_hour, lt.tm_min, lt.tm_sec)


def build_zip(staging_dir: Path, output_zip: Path) -> None:
    if not staging_dir.is_dir():
        raise SystemExit(f"Staging directory not found: {staging_dir}")

    xctestruns = sorted(staging_dir.glob("*.xctestrun"))
    if len(xctestruns) != 1:
        raise SystemExit(
            f"Expected exactly 1 .xctestrun at staging root, found {len(xctestruns)}: {xctestruns}"
        )

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    if output_zip.exists():
        output_zip.unlink()

    entries = []
    for root, dirs, files in os.walk(staging_dir):
        dirs.sort()
        files.sort()
        root_path = Path(root)
        if root_path != staging_dir:
            entries.append((root_path, True))
        for f in files:
            entries.append((root_path / f, False))

    entries.sort(key=lambda item: item[0].relative_to(staging_dir).as_posix() + ("/" if item[1] else ""))

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for fs_path, is_dir in entries:
            rel_posix = fs_path.relative_to(staging_dir).as_posix()
            st = fs_path.lstat()
            dt = zip_datetime_from_mtime(st.st_mtime)

            if fs_path.is_symlink():
                zi = zipfile.ZipInfo(filename=rel_posix, date_time=dt)
                zi.create_system = 3
                zi.compress_type = zipfile.ZIP_STORED
                zi.external_attr = (0o120755 << 16)
                target = os.readlink(fs_path)
                zf.writestr(zi, target.encode("utf-8"))
            elif is_dir:
                arcname = rel_posix.rstrip("/") + "/"
                zi = zipfile.ZipInfo(filename=arcname, date_time=dt)
                zi.create_system = 3
                zi.compress_type = zipfile.ZIP_STORED
                zi.external_attr = (0o040755 << 16) | 0x10
                zf.writestr(zi, b"")
            else:
                exec_flag = is_executable_file(fs_path)
                unix_mode = 0o100755 if exec_flag else 0o100644
                zi = zipfile.ZipInfo(filename=rel_posix, date_time=dt)
                zi.create_system = 3
                zi.compress_type = zipfile.ZIP_DEFLATED
                zi.external_attr = (unix_mode << 16)
                with zf.open(zi, "w") as dest_fh, fs_path.open("rb") as src_fh:
                    while True:
                        chunk = src_fh.read(1024 * 1024)
                        if not chunk:
                            break
                        dest_fh.write(chunk)

    print(f"[package_xctest_zip] Wrote {output_zip} ({output_zip.stat().st_size} bytes)")
    verify_zip(output_zip)


def format_mode(mode: int) -> str:
    return stat.filemode(mode)


def verify_zip(zip_path: Path) -> None:
    if not zip_path.is_file():
        raise SystemExit(f"Zip file does not exist: {zip_path}")

    errors = []
    names_set = set()
    root_xctestruns = []

    print(f"[package_xctest_zip] Verifying {zip_path} ({zip_path.stat().st_size} bytes):")
    with zipfile.ZipFile(zip_path, "r") as zf:
        bad_crc = zf.testzip()
        if bad_crc is not None:
            errors.append(f"CRC check failed on entry: {bad_crc}")

        for zi in zf.infolist():
            names_set.add(zi.filename.rstrip("/"))
            full_mode = (zi.external_attr >> 16) & 0xFFFF
            perm_bits = full_mode & 0o7777
            mode_str = format_mode(full_mode) if full_mode else "??????????"
            print(
                f"  {mode_str} (0o{full_mode:06o}) sys={zi.create_system} "
                f"size={zi.file_size:>10}  {zi.filename}"
            )

            if zi.create_system != 3:
                errors.append(f"Entry {zi.filename} has create_system={zi.create_system} (expected 3/Unix)")

            if zi.filename.endswith("/"):
                if full_mode != 0o040755:
                    errors.append(
                        f"Directory {zi.filename} has mode 0o{full_mode:06o} (expected 0o040755)"
                    )
            else:
                if "/" not in zi.filename and zi.filename.endswith(".xctestrun"):
                    root_xctestruns.append(zi.filename)

                with zf.open(zi, "r") as fh:
                    head = fh.read(8)
                should_be_exec = is_executable_bytes(head)
                expected_mode = 0o100755 if should_be_exec else 0o100644
                if full_mode != expected_mode:
                    errors.append(
                        f"File {zi.filename} has mode 0o{full_mode:06o} "
                        f"(perm 0o{perm_bits:04o}), expected 0o{expected_mode:06o}"
                    )

        if len(root_xctestruns) != 1:
            errors.append(f"Expected 1 .xctestrun at zip root, found: {root_xctestruns}")
        else:
            xctestrun_name = root_xctestruns[0]
            plist_data = plistlib.loads(zf.read(xctestrun_name))
            targets = []
            collect_test_target_dicts(plist_data, targets)
            if not targets:
                errors.append(f"No test targets with TestHostPath/TestBundlePath found in {xctestrun_name}")
            for t in targets:
                host_raw = t["TestHostPath"]
                bundle_raw = t["TestBundlePath"]
                host_rel = host_raw.replace("__TESTROOT__/", "").strip("/")
                bundle_rel = (
                    bundle_raw.replace("__TESTHOST__", host_rel)
                    .replace("__TESTROOT__/", "")
                    .strip("/")
                )
                print(f"  [xctestrun] TestHostPath   -> {host_rel}")
                print(f"  [xctestrun] TestBundlePath -> {bundle_rel}")
                if host_rel not in names_set:
                    errors.append(f"TestHostPath {host_rel} missing from zip")
                if bundle_rel not in names_set:
                    errors.append(f"TestBundlePath {bundle_rel} missing from zip")

        required_entries = [
            "Release-iphoneos/QwenProbeHost.app",
            "Release-iphoneos/QwenProbeHost.app/QwenProbeHost",
            "Release-iphoneos/QwenProbeHost.app/Frameworks/llama.framework/llama",
            "Release-iphoneos/QwenProbeHost.app/PlugIns/QwenProbeTests.xctest",
            "Release-iphoneos/QwenProbeHost.app/PlugIns/QwenProbeTests.xctest/QwenProbeTests",
        ]
        for req in required_entries:
            if req not in names_set:
                errors.append(f"Required entry missing from zip: {req}")

    if errors:
        print("[package_xctest_zip] VERIFICATION FAILED:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        raise SystemExit(1)

    print("[package_xctest_zip] Verification PASSED: layout, .xctestrun paths, and 0755/0644 mode bits OK.")


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(
            "Usage:\n"
            "  package_xctest_zip.py clean-staging <staging_dir>\n"
            "  package_xctest_zip.py normalize-dir <staging_dir>\n"
            "  package_xctest_zip.py sign-dir <staging_dir>\n"
            "  package_xctest_zip.py build-zip <staging_dir> <output_zip>\n"
            "  package_xctest_zip.py verify-zip <output_zip>"
        )

    cmd = sys.argv[1]
    if cmd == "clean-staging":
        clean_staging(Path(sys.argv[2]))
    elif cmd == "normalize-dir":
        normalize_dir(Path(sys.argv[2]))
    elif cmd == "sign-dir":
        sign_dir(Path(sys.argv[2]))
    elif cmd == "build-zip":
        if len(sys.argv) != 4:
            raise SystemExit("Usage: package_xctest_zip.py build-zip <staging_dir> <output_zip>")
        build_zip(Path(sys.argv[2]), Path(sys.argv[3]))
    elif cmd == "verify-zip":
        verify_zip(Path(sys.argv[2]))
    else:
        raise SystemExit(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
