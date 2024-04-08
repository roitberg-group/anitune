#!/bin/bash

# Create conda env and update it with the dependencies of the submodules.
# (maybe it is best to manually synchronize the dependencies?)

_env_name='ani-ftune'

if command -v mamba > /dev/null; then
    _cmd="mamba"
elif command -v conda > /dev/null; then
    _cmd="conda"
else
    echo "Please install mamba, mamba command could not be found."
    exit 1
fi

if [ -d "$HOME/Conda/envs/$_env_name" ]; then
    _scmd=update
else
    _scmd=create
fi

_script_dir="$(dirname "$0")"
"$_cmd $_scmd" -f "${_script_dir}/environment.yaml"

_scmd=update
_submodules_dir="${_script_dir}/submodules"
for d in "${_submodules_dir}/"*; do
    _env_fpath="${d}/environment.yaml"
    if [ -f "${_env_fpath}" ]; then
        "$_cmd" env "$_scmd" --name "${_env_name}" -f "${_env_fpath}"
    fi
done
unset _env_fpath
unset d
unset _submodules_dir
unset _scmd
unset _script_dir
unset _cmd
unset _env_name
echo "Environment created and updated with submodules"
