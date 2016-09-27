#!/usr/bin/env bash

set -e
set -o nounset

basedir=$(cd $(dirname $0)/..;pwd)

function fat_echo() {
    echo "############################################"
    echo "########## $1"
}

function wget_or_curl() {
  [ $# -eq 2 ] || { echo "Usage: wget_or_curl <url> <fpath>" && exit 1; }
  if type wget &> /dev/null; then
    local download_cmd="wget -T 10 -t 3 -O"
  else
    local download_cmd="curl -L -o"
  fi
  $download_cmd "$2" "$1"
}

target="embeddings-original.EMBEDDING_SIZE=25.txt"
if [ ! -e "$basedir/sample/$target" ]; then
    fat_echo "Downloading embeddings from metaoptimize.s3.amazonaws.com"
    (
        wget_or_curl http://metaoptimize.s3.amazonaws.com/cw-embeddings-ACL2010/$target.gz $basedir/sample/$target.gz
        gzip -d $basedir/sample/$target.gz
    )
fi
