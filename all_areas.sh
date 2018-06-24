#!/bin/bash
for i in "$@"; do 
echo "==== $i ===="
./Mapper.py $i;
done
