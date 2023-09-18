#!/bin/bash
echo monitor $1
CHILDREN="$..children[?(@.stat.pid"\ ==\ $1")]"

echo $CHILDREN
cmd="procpath record -i 1 -r 1000 -d infer.sqlite"
echo $cmd
which python
$cmd "$CHILDREN"
