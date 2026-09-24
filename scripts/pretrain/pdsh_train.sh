#!/bin/bash
set -x
# resume train flag, set to be true if using auto resume
AUTO_RESUME_TRAIN=false
# actual train host count
TRAIN_HOST_NUM=0
# hostfile used during auto resume, must be set in the ceph!!!
HOST_PATH=/path/xxxx/hostfile
# if notify by call when exception occures or just notify in wechat
ENABLE_CALL=false
# the experiment owner
OWNER=owner1
# the env file
ENV_FILE=/tmp/resume_train_env.sh 
# wechat robot
WEBHOOK_URL=https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=ac700452-14ee-4a06-aa13-db4c57305833
# monitor log directory
MONITOR_LOG=/tmp/

source scripts/utils/env.sh

if [ $# -ne 1 ]; then
  echo "Usage: $0 <script>"
  exit 1
fi

script=$1
export node_ip=$(echo ${NODE_IP_LIST} | sed 's/:8//g')

if [ ! -d $PROJECT_BASE ]; then
    echo "PROJECT_BASE not exist: $PROJECT_BASE"
    exit 1
fi

CURRENT_DIR=$(pwd)
# worker_num=$(( $NODE_NUM /8 )) # 当超过32台机器时,需要明确指定机器数量
worker_num=$TAIJI_HOST_NUM

pdsh -w $node_ip -f $worker_num "cd ${CURRENT_DIR}; bash $script"
