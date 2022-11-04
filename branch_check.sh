#/bin/bash

if [[ $(uname) == 'Linux' ]]; then
    echo 
    echo "#Setup git proxy"
    source ./scripts/setup_git_proxy.sh 
fi

source ./scripts/git_one_line.sh


