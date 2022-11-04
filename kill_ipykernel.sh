#/bin/bash

ps -ef | grep 'ipykernel_launcher' | grep -v grep | awk '{print $2}' | xargs -r kill -9
ps -ef | grep 'jupyter' | grep -v grep | awk '{print $2}' | xargs -r kill -9

