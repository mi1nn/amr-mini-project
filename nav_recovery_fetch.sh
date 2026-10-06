#!/bin/bash
source /opt/ros/jazzy/setup.bash

readonly STEPS=(
  "controller_server"
  "smoother_server"
  "planner_server"
  "route_server"
  "behavior_server"
  "velocity_smoother"
  "collision_monitor"
  "bt_navigator"
  "waypoint_follower"
  "docking_server"
)


WHITE='\033[1;37m'
L_YELLOW='\033[38;5;229m'
YELLOW='\033[1;33m'
RESET='\033[0m'

clear
echo ""
echo -e "${L_YELLOW}                     XXXXXXXXXX                    ${RESET}"
echo -e "${L_YELLOW}                    X          X                   ${RESET}"
echo -e "${L_YELLOW}                     XXXXXXXXXX                    ${RESET}"
echo "                                                   "
echo "                   XXXXXXXXXXXX                   "
echo "              XXX XX            XX XXXX            "
echo "             X   X                X    X           "
echo "            X                           X          "
echo "    ✨       X    X                  X   X          " 
echo "             XXXX   XXX      XXX     XXX           "
echo -e "${L_YELLOW}    X          ${RESET}X    XXX      XXX     X             "
echo -e "${L_YELLOW}   XXX         ${RESET}X                     X             "
echo -e "${L_YELLOW} XXXXXXXX      ${RESET}X                     X             "
echo -e "${L_YELLOW}  XXXXXX       ${RESET}X         X           X             "
echo -e "${L_YELLOW}  XXX XX       ${RESET}X       XX XX         X             "
echo "     X          XX      XXX        XX    XXXXXXXX  "
echo "      X           XXXXXXXXXXXXXXXXX  XXXXX       X "
echo "       X        XX                 XXX            X"
echo "        X XXXXXX          XXXXX     X       X     X"
echo "         XX              X           XX     XXXXXX    "
echo "          XXXXXX          XXXXXX     X XXXXX       "
echo "               X                     X             "
echo -e "\n${WHITE}ଘ(੭ˊᵕˋ)੭* 🌷${RESET} ${YELLOW}Nav2 Recovery Magic Starting! ${RESET}"


echo -e "\n=========================================="
echo -e "       Nav2 Lifecycle Step List"
echo -e "=========================================="
for i in "${!STEPS[@]}"; do
  printf "  Step [%d]: %s\n" $((i+1)) "${STEPS[$i]}"
done
echo -e "==========================================\n"


while true; do
  while true; do
    read -p "로봇 네임스페이스 숫자 입력 (0-11): " ns || { echo ""; return 0 2>/dev/null || exit 0; }
    if [[ "$ns" =~ ^([0-9]|1[01])$ ]]; then
      break
    else
      echo "오류: 네임스페이스는 0에서 11 사이의 숫자여야 합니다."
    fi
  done
  
  current_robot="robot${ns}"

  while true; do
    while true; do
      read -p "실패한 Step 인덱스 (1-10): " idx || { echo ""; return 0 2>/dev/null || exit 0; }
      if [[ "$idx" =~ ^([1-9]|10)$ ]]; then
        break
      else
        echo "오류: 인덱스는 1에서 10 사이여야 합니다."
      fi
    done

    echo "------------------------------------------"
    echo "현재 대상: $current_robot"
    echo "복구 범위: ${STEPS[$idx-1]} 부터 마지막까지"
    echo "------------------------------------------"

    for (( i=idx-1; i<${#STEPS[@]}; i++ )); do
      current_step=${STEPS[$i]}
      
      echo "[실행] /$current_robot/$current_step 설정 및 활성화 중..."
      
      # 실제 ROS 2 명령어 실행
      ros2 lifecycle set "/$current_robot/$current_step" configure
      ros2 lifecycle set "/$current_robot/$current_step" activate
    done

    echo "------------------------------------------"
    echo "작업이 완료되었습니다."

    # 5. 추가 작업 선택
    read -p "추가 작업 - 추가 복구(a), 로봇 변경(r), 종료(d): " option || { echo ""; return 0 2>/dev/null || exit 0; }

    case $option in
      [aA]) 
        continue 
        ;;
      [rR]) 
        break 
        ;;
      [dD]) 
        echo "스크립트를 종료합니다."
        return 0 2>/dev/null || exit 0
        ;;
      *) 
        echo "잘못된 입력입니다. 종료합니다."
        return 1 2>/dev/null || exit 1
        ;;
    esac
  done
done
