#!/bin/bash

for i in {00..10};
do
    FF=experiment-$i.sh
    echo Running $FF
    chmod +x ./$FF
    ./$FF
done


