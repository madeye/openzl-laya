#!/usr/bin/env bash
# Render the Homebrew formula for a published OpenZL-Laya release into a
# checkout of https://github.com/madeye/homebrew-tap.
#
# Usage: scripts/laya/update_homebrew_formula.sh VERSION SHA256 [TAP_DIR]
#   VERSION  release version without the leading v, e.g. 0.2.5-laya.1
#   SHA256   sha256 of openzl-laya-VERSION-macos-arm64.tar.gz (from the
#            release notes or its .sha256 asset)
#   TAP_DIR  tap checkout (default: ../homebrew-tap next to this repository)
set -euo pipefail

if [ $# -lt 2 ]; then
    sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
    exit 2
fi
version=$1
sha256=$2
root=$(cd "$(dirname "$0")/../.." && pwd)
tap=${3:-"$root/../homebrew-tap"}

[[ $version =~ ^[0-9]+\.[0-9]+\.[0-9]+-laya\.[0-9]+$ ]] \
    || { echo "unexpected version: $version" >&2; exit 1; }
[[ $sha256 =~ ^[0-9a-f]{64}$ ]] \
    || { echo "unexpected sha256: $sha256" >&2; exit 1; }
[ -d "$tap" ] || { echo "no tap checkout at $tap" >&2; exit 1; }

mkdir -p "$tap/Formula"
sed -e "s/@VERSION@/$version/g" -e "s/@SHA256@/$sha256/g" \
    "$root/packaging/laya/openzl-laya.rb.in" > "$tap/Formula/openzl-laya.rb"
echo "wrote $tap/Formula/openzl-laya.rb for $version"
echo "next: cd $tap && git commit -am 'openzl-laya $version' && git push"
