#!/bin/bash

pysource() {
    #echo $(python -c "import ${1}; import inspect; print(inspect.getsourcefile(${1}))")
    echo $(python -c "import os; import ${1}; print(os.path.dirname(${1}.__file__))")
}

L5KIT_SOURCE=$(pysource l5kit)
echo "Found l5kit at $L5KIT_SOURCE"
patch -ruN  -b --verbose  -i l5kit.patch "$L5KIT_SOURCE/dataset/ego.py"

