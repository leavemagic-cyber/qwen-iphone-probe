#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LLAMA_TAG="b12020"
LLAMA_ZIP_NAME="llama-${LLAMA_TAG}-xcframework.zip"
LLAMA_ZIP_URL="https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_TAG}/${LLAMA_ZIP_NAME}"
LLAMA_ZIP_SHA256="78fdf6ec32fad1f0e55a3fd78401841ba0a7b34f26b30078c6dc8d7951d81ae5"

echo "=== [1/7] Fetching llama.cpp prebuilt xcframework (${LLAMA_TAG}) ==="
mkdir -p .cache
if [[ ! -f ".cache/${LLAMA_ZIP_NAME}" ]]; then
  curl -fL --retry 3 --retry-delay 3 -o ".cache/${LLAMA_ZIP_NAME}" "${LLAMA_ZIP_URL}"
fi

ACTUAL_SHA256="$(shasum -a 256 ".cache/${LLAMA_ZIP_NAME}" | awk '{print $1}')"
if [[ "${ACTUAL_SHA256}" != "${LLAMA_ZIP_SHA256}" ]]; then
  echo "SHA-256 mismatch for ${LLAMA_ZIP_NAME}: got ${ACTUAL_SHA256}, expected ${LLAMA_ZIP_SHA256}" >&2
  exit 1
fi
echo "Verified ${LLAMA_ZIP_NAME} SHA-256: ${ACTUAL_SHA256}"

rm -rf build-apple
unzip -q ".cache/${LLAMA_ZIP_NAME}" -d .

XCFRAMEWORK_IOS_FW="build-apple/llama.xcframework/ios-arm64/llama.framework"
if [[ ! -f "${XCFRAMEWORK_IOS_FW}/Headers/mtmd.h" ]]; then
  echo "ERROR: mtmd.h missing in ${XCFRAMEWORK_IOS_FW}/Headers" >&2
  exit 1
fi
nm -gU "${XCFRAMEWORK_IOS_FW}/llama" > .cache/nm_llama.txt
if ! grep -F '_mtmd_init_from_file' .cache/nm_llama.txt >/dev/null; then
  echo "ERROR: _mtmd_init_from_file symbol missing in ${XCFRAMEWORK_IOS_FW}/llama" >&2
  exit 1
fi
echo "Verified mtmd headers and arm64 symbols in ${XCFRAMEWORK_IOS_FW}"

echo "=== [2/7] Generating Xcode project with XcodeGen ==="
rm -rf QwenIphoneProbe.xcodeproj
xcodegen generate

echo "=== [3/7] Running xcodebuild build-for-testing (Release, iphoneos) ==="
rm -rf build staging dist
xcodebuild build-for-testing \
  -project QwenIphoneProbe.xcodeproj \
  -scheme QwenProbeHost \
  -configuration Release \
  -sdk iphoneos \
  -destination "generic/platform=iOS" \
  -derivedDataPath build \
  CODE_SIGNING_ALLOWED=NO \
  CODE_SIGNING_REQUIRED=NO \
  CODE_SIGN_IDENTITY="" \
  DEVELOPMENT_TEAM=""

echo "=== [4/7] Staging Release-iphoneos and .xctestrun ==="
PRODUCTS_DIR="build/Build/Products"
shopt -s nullglob
XCTESTRUN_FILES=("${PRODUCTS_DIR}"/*.xctestrun)
shopt -u nullglob

if [[ "${#XCTESTRUN_FILES[@]}" -ne 1 ]]; then
  echo "ERROR: Expected exactly 1 .xctestrun file in ${PRODUCTS_DIR}, found ${#XCTESTRUN_FILES[@]}" >&2
  ls -la "${PRODUCTS_DIR}" >&2 || true
  exit 1
fi

mkdir -p staging
cp "${XCTESTRUN_FILES[0]}" staging/
ditto "${PRODUCTS_DIR}/Release-iphoneos" "staging/Release-iphoneos"

rm -rf "staging/Release-iphoneos/XCFrameworkIntermediates"
find staging -name "*.dSYM" -prune -exec rm -rf {} +
find staging -name "*.swiftmodule" -prune -exec rm -rf {} +
find staging -name ".DS_Store" -delete

APP_PATH="staging/Release-iphoneos/QwenProbeHost.app"
EMBEDDED_FW="${APP_PATH}/Frameworks/llama.framework"
TEST_BUNDLE="${APP_PATH}/PlugIns/QwenProbeTests.xctest"

if [[ ! -d "${EMBEDDED_FW}" ]]; then
  echo "ERROR: Expected embedded framework at ${EMBEDDED_FW}" >&2
  find staging -maxdepth 5 >&2
  exit 1
fi
if [[ ! -d "${TEST_BUNDLE}" ]]; then
  echo "ERROR: Expected app-hosted test bundle at ${TEST_BUNDLE}" >&2
  find staging -maxdepth 5 >&2
  exit 1
fi
nm "${TEST_BUNDLE}/QwenProbeTests" > .cache/nm_tests.txt
if ! grep -F 'testQwenOnDevice' .cache/nm_tests.txt; then
  echo "ERROR: testQwenOnDevice symbol not found in ${TEST_BUNDLE}/QwenProbeTests" >&2
  exit 1
fi
echo "Verified testQwenOnDevice symbol in ${TEST_BUNDLE}/QwenProbeTests"

echo "=== [5/7] Normalizing mode bits & ad-hoc signing inside-out ==="
python3 scripts/package_xctest_zip.py normalize-dir staging

shopt -s nullglob
for item in "${TEST_BUNDLE}/Frameworks/"*.dylib "${TEST_BUNDLE}/Frameworks/"*.framework; do
  echo "Ad-hoc signing nested test dependency: ${item}"
  codesign --force --sign - --timestamp=none "${item}"
done

for item in "${APP_PATH}/Frameworks/"*.dylib "${APP_PATH}/Frameworks/"*.framework; do
  echo "Ad-hoc signing embedded app framework/dylib: ${item}"
  codesign --force --sign - --timestamp=none "${item}"
done

for item in "${APP_PATH}/PlugIns/"*.xctest; do
  echo "Ad-hoc signing PlugIns test bundle: ${item}"
  codesign --force --sign - --timestamp=none "${item}"
done

for item in staging/Release-iphoneos/*.dylib staging/Release-iphoneos/*.framework staging/Release-iphoneos/*.xctest; do
  echo "Ad-hoc signing top-level product: ${item}"
  codesign --force --sign - --timestamp=none "${item}"
done
shopt -u nullglob

echo "Ad-hoc signing host app: ${APP_PATH}"
codesign --force --sign - --timestamp=none "${APP_PATH}"

python3 scripts/package_xctest_zip.py normalize-dir staging

echo "=== [6/7] Verifying ad-hoc signatures (codesign --verify --deep) ==="
codesign --verify --deep --strict --verbose=4 "${EMBEDDED_FW}"
codesign --verify --deep --strict --verbose=4 "${TEST_BUNDLE}"
codesign --verify --deep --strict --verbose=4 "${APP_PATH}"
codesign -dv --verbose=4 "${APP_PATH}" 2>&1

echo "=== [7/7] Packaging & verifying Firebase Test Lab XCTest zip ==="
mkdir -p dist
ZIP_OUT="dist/qwen-iphone-probe-xctest.zip"
python3 scripts/package_xctest_zip.py build-zip staging "${ZIP_OUT}"

echo "=== zipinfo -l ${ZIP_OUT} ==="
zipinfo -l "${ZIP_OUT}"

VERIFY_DIR="$(mktemp -d)"
trap 'rm -rf "${VERIFY_DIR}"' EXIT
ditto -x -k "${ZIP_OUT}" "${VERIFY_DIR}"

echo "=== Verifying codesign --verify --deep after ditto unzip round-trip ==="
codesign --verify --deep --strict --verbose=4 "${VERIFY_DIR}/Release-iphoneos/QwenProbeHost.app/Frameworks/llama.framework"
codesign --verify --deep --strict --verbose=4 "${VERIFY_DIR}/Release-iphoneos/QwenProbeHost.app/PlugIns/QwenProbeTests.xctest"
codesign --verify --deep --strict --verbose=4 "${VERIFY_DIR}/Release-iphoneos/QwenProbeHost.app"
echo "=== BUILD AND PACKAGE SUCCEEDED: ${ZIP_OUT} ==="
