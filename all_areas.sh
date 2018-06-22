#!/bin/bash
for i in $1; do 
echo "==== $i ===="
if ./Mapper.py $i; then
: ; else 
break 
fi
done
