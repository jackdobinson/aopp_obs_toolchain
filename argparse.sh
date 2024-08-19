#!/usr/bin/env bash

# Turn on "strict" mode
set -o errexit -o nounset -o pipefail
IFS=$'\n\t'

ARG_OBFUSCATION=${ARG_OBFUSCATION:-TEMP_ARG_84767488}
USAGE_OBFUSCATION=${USAGE_OBFUSCATION:-TEMP_USAGE_84767487}
SHOULD_EXIT_FLAG="0"
function print_usage_and_exit() {
	print_usage
	exit 0
}

function print_usage() {

	declare -n USAGE_INFO=${USAGE_OBFUSCATION}
	declare -n USAGE_POS_ARGS=${USAGE_OBFUSCATION}_POS_ARGS
	declare -n USAGE_KEY_ARGS=${USAGE_OBFUSCATION}_KEY_ARGS
	
	USAGE_STR=(
		${USAGE_INFO['SCRIPT_NAME']}
	)
	
	DESCRIPTION_STR=(
		"${USAGE_INFO['SCRIPT_DESCRIPTION']}"
		" "
		" "
		"Positional Arguments:"
		" "
	)
	
	
	for ARG_INFO_NAME in "${USAGE_POS_ARGS[@]}"; do
		#echo "POSITIONAL ARGS: ${ARG_INFO_NAME}"
		local ARG_INFO_ARRAY_NAME="${ARG_INFO_NAME}['NAME']"
		local ARG_INFO_ARRAY_TYPE="${ARG_INFO_NAME}['TYPE']"
		local ARG_INFO_ARRAY_DESCRIPTION="${ARG_INFO_NAME}['DESCRIPTION']"
		local ARG_INFO_ARRAY_ACTION="${ARG_INFO_NAME}['ACTION']"
		local ARG_INFO_ARRAY_DEFAULT="${ARG_INFO_NAME}['DEFAULT']"
		local ARG_INFO_ARRAY_DOMAIN="${ARG_INFO_NAME}['DOMAIN']"
		#echo "${!ARG_INFO_ARRAY_NAME} ${!ARG_INFO_ARRAY_TYPE} ${!ARG_INFO_ARRAY_DESCRIPTION} ${!ARG_INFO_ARRAY_ACTION} ${!ARG_INFO_ARRAY_DEFAULT}"
		
		local DOMAIN_INFO_STR=""
		if [ -n "${!ARG_INFO_ARRAY_DOMAIN}" ]; then
			local DOMAIN_STR=${!ARG_INFO_ARRAY_DOMAIN}
			local DOMAIN_ARRAY=()
			local SEP='|'
			local CHAR
			local ACC
			for ((I = 0; I < ${#DOMAIN_STR}; I++)); do
				CHAR=${DOMAIN_STR:$I:1}
				#echo "$CHAR"
				if [ "$CHAR" = "$SEP" ]; then
					DOMAIN_ARRAY+=($ACC)
					ACC=""
				else
					ACC+=$CHAR
				fi
				if [ $I -eq $((${#DOMAIN_STR}-1)) ]; then
					DOMAIN_ARRAY+=($ACC)
				fi
			done
			#echo "${DOMAIN_ARRAY[@]}"
			DOMAIN_INFO_STR+="{"
			for IDX in ${!DOMAIN_ARRAY[@]}; do
				#echo "XXXXX: $IDX ## ${#DOMAIN_ARRAY[@]} ## ${DOMAIN_ARRAY[$IDX]}"
				if [ "$IDX" != "$((${#DOMAIN_ARRAY[@]}-1))" ]; then
					DOMAIN_INFO_STR+="${DOMAIN_ARRAY[$IDX]}|"
				else 
					DOMAIN_INFO_STR+="${DOMAIN_ARRAY[$IDX]}"
				fi
			done
			DOMAIN_INFO_STR+="}"
		fi
	
	
		#USAGE_STR+=("<${!ARG_INFO_ARRAY_TYPE}>")
		SHORT_STR="<${!ARG_INFO_ARRAY_TYPE}"
		if [ -n "$DOMAIN_INFO_STR" ]; then
			SHORT_STR+=" $DOMAIN_INFO_STR"
		fi
		SHORT_STR+=">"
	
	
		local INFO_STR=" ${!ARG_INFO_ARRAY_NAME}"
		if [ ! "${!ARG_INFO_ARRAY_TYPE}" = "flag" ]; then
			INFO_STR+=" : <${!ARG_INFO_ARRAY_TYPE}"
			if [ -n "$DOMAIN_INFO_STR" ]; then
				INFO_STR+=" $DOMAIN_INFO_STR"
			fi
			INFO_STR+=">"
			
		fi
		if [ -n "${!ARG_INFO_ARRAY_DEFAULT}" ]; then
			INFO_STR+=" = ${!ARG_INFO_ARRAY_DEFAULT}"
		fi
	
		USAGE_STR+=("$SHORT_STR")
	
		DESCRIPTION_STR+=("$INFO_STR")
		DESCRIPTION_STR+=("    ${!ARG_INFO_ARRAY_DESCRIPTION}\n")
	
	done
	
	DESCRIPTION_STR+=(
		" "
		'Keyword Arguments:'
		" "
	)
		
	for ARG_INFO_NAME in ${USAGE_KEY_ARGS[@]}; do
		#echo "KEYWORD ARGS: ${ARG_INFO_NAME}"
		local ARG_INFO_ARRAY_NAME="${ARG_INFO_NAME}['NAME']"
		local ARG_INFO_ARRAY_TYPE="${ARG_INFO_NAME}['TYPE']"
		local ARG_INFO_ARRAY_DESCRIPTION="${ARG_INFO_NAME}['DESCRIPTION']"
		local ARG_INFO_ARRAY_ACTION="${ARG_INFO_NAME}['ACTION']"
		local ARG_INFO_ARRAY_DEFAULT="${ARG_INFO_NAME}['DEFAULT']"
		local ARG_INFO_ARRAY_DOMAIN="${ARG_INFO_NAME}['DOMAIN']"
		#echo "'${!ARG_INFO_ARRAY_NAME}' '${!ARG_INFO_ARRAY_TYPE}' '${!ARG_INFO_ARRAY_DESCRIPTION}' '${!ARG_INFO_ARRAY_ACTION}' '${!ARG_INFO_ARRAY_DEFAULT}'"

		local DOMAIN_INFO_STR=""
		if [ -n "${!ARG_INFO_ARRAY_DOMAIN}" ]; then
			local DOMAIN_STR=${!ARG_INFO_ARRAY_DOMAIN}
			local DOMAIN_ARRAY=()
			local SEP='|'
			local CHAR=""
			local ACC=""
			for ((I = 0; I < ${#DOMAIN_STR}; I++)); do
				CHAR=${DOMAIN_STR:$I:1}
				#echo "$CHAR"
				if [ "$CHAR" = "$SEP" ]; then
					DOMAIN_ARRAY+=($ACC)
					ACC=""
				else
					ACC+=$CHAR
				fi
				if [ $I -eq $((${#DOMAIN_STR}-1)) ]; then
					DOMAIN_ARRAY+=($ACC)
				fi
			done
			#echo "${DOMAIN_ARRAY[@]}"
			DOMAIN_INFO_STR+="{"
			for IDX in ${!DOMAIN_ARRAY[@]}; do
				#echo "XXXXX: $IDX ## ${#DOMAIN_ARRAY[@]} ## ${DOMAIN_ARRAY[$IDX]}"
				if [ "$IDX" != "$((${#DOMAIN_ARRAY[@]}-1))" ]; then
					DOMAIN_INFO_STR+="${DOMAIN_ARRAY[$IDX]}|"
				else 
					DOMAIN_INFO_STR+="${DOMAIN_ARRAY[$IDX]}"
				fi
			done
			DOMAIN_INFO_STR+="}"
		fi


		local SHORT_STR="[${!ARG_INFO_ARRAY_NAME}"		
		local INFO_STR=" ${!ARG_INFO_ARRAY_NAME}"
		
		if [ ! "${!ARG_INFO_ARRAY_TYPE}" = "flag" ]; then
			INFO_STR+=" : <${!ARG_INFO_ARRAY_TYPE}"
			SHORT_STR+=" <${!ARG_INFO_ARRAY_TYPE}"
			if [ -n "$DOMAIN_INFO_STR" ]; then
				INFO_STR+=" $DOMAIN_INFO_STR"
				SHORT_STR+=" $DOMAIN_INFO_STR"
			fi
			INFO_STR+=">"
			SHORT_STR+=">"
		fi
		if [ -n "${!ARG_INFO_ARRAY_DEFAULT}" ]; then
			INFO_STR+=" = ${!ARG_INFO_ARRAY_DEFAULT}"
		fi
		
		USAGE_STR+=("${SHORT_STR}]")
		
		DESCRIPTION_STR+=("$INFO_STR")
		DESCRIPTION_STR+=("    ${!ARG_INFO_ARRAY_DESCRIPTION}\n")
	
	done
	
	echo "USAGE:"
	echo -e "  ${USAGE_STR[@]}\n"
	for X in ${DESCRIPTION_STR[@]}; do
		echo -e "  ${X}"
	done
	
}


function set_usage_info() {

	declare -Ag ${USAGE_OBFUSCATION}
	declare -n USAGE_INFO=${USAGE_OBFUSCATION}

	USAGE_INFO['SCRIPT_NAME']=$1
	USAGE_INFO['SCRIPT_DESCRIPTION']=$2
	
	
	#echo "USAGE_INFO:"
	#for X in ${!USAGE_INFO[@]}; do
	#	echo "    $X=${USAGE_INFO[$X]}"
	#done
}



function argparse() {

	#echo "In 'argparse'"
	#echo "Arguments $@"
	
	# Loop over possible ARG_INFO_* names
	#while declare -p "${ARG_INFO_NAME}" &>/dev/null; do
	#done
	
	declare -gA ARGS
	declare -ga ${USAGE_OBFUSCATION}_POS_ARGS
	declare -ga ${USAGE_OBFUSCATION}_KEY_ARGS
	declare -n USAGE_INFO=${USAGE_OBFUSCATION}
	declare -n USAGE_POS_ARGS=${USAGE_OBFUSCATION}_POS_ARGS
	declare -n USAGE_KEY_ARGS=${USAGE_OBFUSCATION}_KEY_ARGS
	
	local POS_ARGS=()
	local POS_ARGS_NAMES=()
	local KEY_ARGS=()
	local KEY_ARGS_NAMES=()
	
	# Get keyword and positonal args
	for ARG_INFO_NAME in ${!TEMP_ARG_*}; do
		#echo "$ARG_INFO_NAME"
		local ARG_INFO_ARRAY="${ARG_INFO_NAME}[@]"
		local ARG_INFO_ARRAY_NAME="${ARG_INFO_NAME}['NAME']"
		local ARG_INFO_ARRAY_TYPE="${ARG_INFO_NAME}['TYPE']"
		local ARG_INFO_ARRAY_DESCRIPTION="${ARG_INFO_NAME}['DESCRIPTION']"
		local ARG_INFO_ARRAY_ACTION="${ARG_INFO_NAME}['ACTION']"
		local ARG_INFO_ARRAY_DEFAULT="${ARG_INFO_NAME}['DEFAULT']"
		
		#echo "${ARG_INFO_NAME}=${!ARG_INFO_ARRAY}"
		
		local P_NAME="${!ARG_INFO_ARRAY_NAME}"
		#echo "P_NAME:0:2=${P_NAME:0:2}"
		case ${P_NAME:0:1} in
			"-")
				#echo "KEY ${P_NAME}"
				KEY_ARGS+=($P_NAME)
				KEY_ARGS_NAMES+=(${ARG_INFO_NAME})
				;;
			*)
				#echo "POS ${P_NAME}"
				POS_ARGS+=($P_NAME)
				POS_ARGS_NAMES+=(${ARG_INFO_NAME})
				;;
		esac
	done
	
	#echo "POS_ARGS=${POS_ARGS[@]}"
	#echo "KEY_ARGS=${KEY_ARGS[@]}"
	#echo "POS_ARGS_NAMES=${POS_ARGS_NAMES[@]}"
	#echo "KEY_ARGS_NAMES=${KEY_ARGS_NAMES[@]}"
	
	
	USAGE_POS_ARGS=(${POS_ARGS_NAMES[@]})
	USAGE_KEY_ARGS=(${KEY_ARGS_NAMES[@]})
	
	
	# Loop over arguments and parse them
	
	local _ARGS=($@)
	local _SKIP_OVER_IDX=-1
	local _N_POS_ARGS_FOUND=0
	for ARG_IDX in ${!_ARGS[@]}; do
	
		#echo "ARG_IDX=${ARG_IDX}"
		#echo "_ARGS[${ARG_IDX}]=${_ARGS[${ARG_IDX}]}"
		#echo "_SKIP_OVER_IDX=$_SKIP_OVER_IDX"
		if [ $ARG_IDX -le $_SKIP_OVER_IDX ]; then
			continue
		fi
	
	
		local _ARG=${_ARGS[$ARG_IDX]}
		local KEY_ARG_FOUND=0
		
		KEY_ARG_IDX=0
		for KEY_ARG in ${KEY_ARGS[@]}; do
			if [ "$_ARG" = "$KEY_ARG" ]; then
				#echo "KEYWORD ARGUMENT"
				KEY_ARG_FOUND=1
				
				local ARG_INFO_NAME=${KEY_ARGS_NAMES[$KEY_ARG_IDX]}
				local ARG_INFO_ARRAY_ACTION="${ARG_INFO_NAME}['ACTION']"
				local ARG_INFO_ARRAY_TYPE="${ARG_INFO_NAME}['TYPE']"
				#echo "${ARG_INFO_ARRAY_ACTION}=${!ARG_INFO_ARRAY_ACTION}"
				
				local VALUE=""
				
				case "${!ARG_INFO_ARRAY_TYPE}" in
					"flag")
						VALUE="$_ARG"
						if [ -n "${!ARG_INFO_ARRAY_ACTION}" ]; then
							${!ARG_INFO_ARRAY_ACTION}
						fi
						;;
					*)
						_SKIP_OVER_IDX=$((${ARG_IDX}+1))
						VALUE=${_ARGS[_SKIP_OVER_IDX]}
						;;
				esac
				
				if [ -n "${!ARG_INFO_ARRAY_ACTION}" ]; then
					#echo "PERFORMING ACTION"
					VALUE=$(${!ARG_INFO_ARRAY_ACTION} ${VALUE})
				fi
				ARGS["${_ARG}"]=${VALUE}
				
				
				continue
			fi
			KEY_ARG_IDX=$((KEY_ARG_IDX+1))
		done
		
		
		if [ "$KEY_ARG_FOUND" = "1" ]; then
			continue
		fi
		
		#echo "MUST BE POS ARG ${_N_POS_ARGS_FOUND}"
		echo "_ARG = ${_ARG}"
		echo "_N_POS_ARGS_FOUND = ${_N_POS_ARGS_FOUND}"
		echo "POS_ARGS[${_N_POS_ARGS_FOUND}] = ${POS_ARGS[${_N_POS_ARGS_FOUND}]}"
		
		ARGS["${POS_ARGS[${_N_POS_ARGS_FOUND}]}"]=${_ARG}
		_N_POS_ARGS_FOUND=$(($_N_POS_ARGS_FOUND+1))
			
	
	done
	
	for IDX in ${!KEY_ARGS[@]}; do
		#echo "${KEY_ARGS[IDX]}"
		if [ ! "${ARGS["${KEY_ARGS[$IDX]}"]+x}" ]; then
			#echo "${KEY_ARGS[IDX]}"
		
			local ARG_INFO_NAME=${KEY_ARGS_NAMES[$IDX]}
			local ARG_INFO_ARRAY_DEFAULT="${ARG_INFO_NAME}['DEFAULT']"
			ARGS[${KEY_ARGS[$IDX]}]=${!ARG_INFO_ARRAY_DEFAULT}
		fi
	done
	
	#echo "ARGS:"
	#for X in ${!ARGS[@]}; do
	#	echo "    $X=${ARGS[$X]}"
	#done
	
	
}

function to_int() {
	echo "$(($1+0))"
}

function add_argument() {
	N_ARGS=${N_ARGS:-0}
	
	#local NEXT_ARG_INFO_NAME="ARG_INFO_${N_ARGS}"	
	local NEXT_ARG_INFO_NAME=$(printf "%s_%05d" "${ARG_OBFUSCATION}" "${N_ARGS}")
	
	# Create a new global variable with desired name	
	declare -Ag ${NEXT_ARG_INFO_NAME}
	
	# Create a local reference to the global variable
	declare -n NEXT_ARG_INFO_REF="${NEXT_ARG_INFO_NAME}"
	
	NEXT_ARG_INFO_REF["NAME"]="$1"
	NEXT_ARG_INFO_REF["TYPE"]="$2"
	NEXT_ARG_INFO_REF["DESCRIPTION"]="$3"
	NEXT_ARG_INFO_REF["ACTION"]="${4:-}"
	NEXT_ARG_INFO_REF["DEFAULT"]="${5:-}"
	NEXT_ARG_INFO_REF["DOMAIN"]="${6:-}"
	
	
	#echo "${NEXT_ARG_INFO_NAME}=${!NEXT_ARG_INFO_REF[@]}"
	#echo "${NEXT_ARG_INFO_NAME}=${NEXT_ARG_INFO_REF[@]}"
	
	#echo "${NEXT_ARG_INFO_NAME}=${!NEXT_ARG_INFO_ARRAY}"
	N_ARGS=$((N_ARGS+1))
}


function test_argparse() {

	set_usage_info "test_script" "description of test script for usage message"
	
	add_argument name0 int description0 to_int
	add_argument -name1 int description1 to_int '' '11|15|0098'
	add_argument --name2 int description2 to_int 99
	add_argument name3 int description3 to_int 0 '0|9|8|7'
	add_argument -h flag 'display help message' print_usage_and_exit
	add_argument -q float 'display help message' '' '' '0->inf'

	TEST_ARGS=(-h 1 2 -name1 5 --name2 'fifty five')
	#for X in ${TEST_ARGS[@]}; do
	#	echo "$X"
	#done

	argparse ${TEST_ARGS[@]}


}

# DEBUGGING
#test_argparse
