# Install CosDATA
set -x

export node_ip=$(echo ${NODE_IP_LIST} | sed 's/:8//g')
CURRENT_DIR=$(pwd)

echo "Current directory: ${CURRENT_DIR}, TAIJI_TOKEN=${TAIJI_TOKEN}"

# # ffmpeg
pdsh -f 512 -w $node_ip "yum --enablerepo=TencentOS-testing install ffmpeg -y"

# audiotools wheel
pdsh -f 512 -w $node_ip "pip install descript_audiotools"

