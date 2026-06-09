#!/bin/bash

# run from the script directory
cd "$(dirname "$0")"

echo "$(dirname "$0")"

replaceable=0
port=0
scorepolicy=0
env=0
seed="NONE"
performanceDir="NONE"
runDir="run"
maxMem="4G"

while [ $# -gt 0 ]
do
    case "$1" in
        -replaceable) replaceable=1;;
        -port) port="$2"; shift;;
        -seed) seed="$2"; shift;;
        -scorepolicy) scorepolicy="$2"; shift;;
        -env) env=1;;
        -runDir) runDir="$2"; shift;;
        -performanceDir) performanceDir="$2"; shift;;
        -maxMem) maxMem="$2"; shift;;
        *) echo >&2 \
            "usage: $0 [-replaceable] [-port 10000] [-seed 123123] [-scorepolicy 0123] [-env] [-runDir /home/asdasd] [-performanceDir /home/asdasd]"
            exit 1;;
    esac
    shift
done
  
if ! [[ $port =~ ^-?[0-9]+$ ]]; then
    echo "Port value should be numeric"
    exit 1
fi


if [ \( $port -lt 0 \) -o \( $port -gt 65535 \) ]; then
    echo "Port value out of range 0-65535"
    exit 1
fi

if ! [[ $scorepolicy =~ ^-?[0-9]+$ ]]; then
    echo "Score policy should be numeric"
    exit 1
fi

fatjar=build/libs/mcprec-6.13.jar
java -Xmx$maxMem \
    -DMINERL_PERSISTENT_SERVER="${MINERL_PERSISTENT_SERVER:-true}" \
    -DMINERL_LAN_ENABLED="${MINERL_LAN_ENABLED:-true}" \
    -DMINERL_LAN_PORT="${MINERL_LAN_PORT:-25565}" \
    -jar $fatjar --envPort=$port
[ $replaceable -gt 0 ]
