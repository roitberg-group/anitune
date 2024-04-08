#!/bin/bash

_script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

if ! command -v pip > /dev/null; then
    echo "Please install pip, command pip could not be found"
    exit 1
fi

if [ -n "$1" ]; then
    _env_name="$1"  # for forwarding
else
    _env_name='ani-ftune'
fi

# Installation for editable conda packanges
if [ -f "$HOME/Conda/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    . "$HOME/Conda/etc/profile.d/conda.sh"
    if [ -f "$HOME/Conda/etc/profile.d/mamba.sh" ]; then
        # shellcheck disable=SC1091
        . "$HOME/Conda/etc/profile.d/mamba.sh"
        mamba activate "${_env_name}"
    else
        conda activate "${_env_name}"
    fi
fi

pip install \
    --editable \
    "${_script_dir}"

# recursively install submodules
_submodules_dir="${_script_dir}/submodules"
for d in "${_submodules_dir}/"*; do
    _install_fpath="${d}/install.sh"
    if [ -f "${_install_fpath}" ]; then
        bash "${_install_fpath}" "${_env_name}"
    fi
done
unset _install_fpath
unset d
unset _submodules_dir
