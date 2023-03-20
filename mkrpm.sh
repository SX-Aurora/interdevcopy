#!/bin/sh

function usage {
  cat <<EOM
Usage: $(basename "$0") [OPTION]...
  -h          Display help
  -r          Assign 0.datetime to release
EOM

  exit 2
}

config_option=" --prefix=/opt/nec/interdevcopy \
    --libdir=/opt/nec/interdevcopy/lib64 \
    --with-veo=/opt/nec/ve/veos --with-cuda=/usr/local/cuda"

while getopts :rh optKey; do
  case "$optKey" in
    r)
      config_option+=" --with-release=0.$(date +%Y%m%d%H%M)"
      ;;
    '-h'|*)
      usage
      ;;
  esac
done

./bootstrap
srcdir=$(pwd)
workdir=$(mktemp -d)

build_status=1
cleanup() {
  if [ $build_status = 0 ]; then
    rm -rf "$workdir"
  fi
}
trap cleanup EXIT

cd "$workdir" && \
"${srcdir}/configure"$config_option &&
make rpm
build_status=$?

