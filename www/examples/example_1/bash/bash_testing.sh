
declare -a LOG_LEVEL_DO_EXIT=('NEVER')
declare -A LOG_LEVEL_VALUES=(['UNSET']=0 ['DEBUG']=10 ['INFO']=20 ['WARN']=30 ['ERROR']=50 ['CRIT']=60 ['TOP']=100 ['NEVER']=101 )
declare -a LOG_LEVEL=('WARN')
function log {
	local LEVEL=$1
	shift 1
	if [ ${LOG_LEVEL_VALUES[${LOG_LEVEL[-1]}]:-0} -le ${LOG_LEVEL_VALUES[${LEVEL}]:-100} ]; then
		echo "${BASH_SOURCE[1]} ${BASH_LINENO[0]} ${FUNCNAME[1]} ${LEVEL}: ${@}"
	fi
	#if [ ${LOG_LEVEL_VALUES[${LOG_LEVEL_DO_EXIT[-1]}]:-0} -le ${LOG_LEVEL_VALUES[${LEVEL}]:-100} ]; then
	#	exit 1
	#fi
}

function trap_func {
	# trap_func <func> <signal> [<signal>...]
	# Passes trapped signal to <trap_func>
	# so we can log it later.
	local FUNC=$1
	shift 1
	local SIGNALS=($@)
	for SIG in ${SIGNALS[@]}; do
		trap "$FUNC ${SIG}" ${SIG}
	done
	log 'DEBUG' "Function \"${FUNC}\" traps signals: ${SIGNALS[@]}"
}


declare -A TRAP_RETURN_CMD_STACK
function trap_return_cmd_stack_print {
	for KEY in ${!TRAP_RETURN_CMD_STACK[@]}; do
		echo "${KEY}: \$(${TRAP_RETURN_CMD_STACK[$KEY]})"
	done
}
function trap_return_cmd_stack_add {
	# arguments: <code>
	# returns : int command_handle
	echo $FUNCNAME
	local HDL=$RANDOM
	
	# Regenerate handle if it already exists
	while [ ${TRAP_RETURN_CMD_STACK[${HDL}]--1} -ge 0 ]; do
		HDL=${RANDOM}
	done
	
	TRAP_RETURN_CMD_STACK[$HDL]=$1
	trap_return_cmd_stack_print
	echo "$HDL"
}
function trap_return_cmd_stack_remove {
	unset TRAP_RETURN_CMD_STACK[$1]
}
function trap_return_cmd_stack_init {
	echo "${FUNCNAME}"
	local TRAP_RETURN_CMD=$(trap -p RETURN)
	local NEW_CMD="trap_return_cmd_stack_run;"
	# If we already have the command in the trap, return
	case $TRAP_RETURN_CMD in
		*${NEW_CMD}* )
			return 0
			;;
	esac 
	# Otherwise add the command
	trap "${NEW_CMD} ${TRAP_RETURN_CMD}" RETURN
}
function trap_return_cmd_stack_run {
	#trap_return_cmd_stack_print
	for CMD in ${TRAP_RETURN_CMD_STACK[@]}; do
		eval "${CMD}"
	done
}

trap_return_cmd_stack_init # initialise

trap_return_cmd_stack_add 'echo "RETURNING..."'
trap_return_cmd_stack_print
echo "$(trap -p RETURN)"

function say_hello_there {
	echo "HELLO"
	echo "--: $(trap -p RETURN)"
	echo "THERE"
}

#declare -f -t say_hello_there
declare -f -t "say_hello_there"

say_hello_there
echo "AFTER"
echo "--: $(trap -p RETURN)"

function run_at_stack_level {
	# <level> <code> [<code>...]
	local TRIGGER_STACK_LEVEL=$1
	shift 1
	local COMMANDS=($@)
	local CURRENT_STACK_LEVEL=$((${#FUNCNAME[@]}-1)) # one less as we don't want to count this function
	if [ $CURRENT_STACK_LEVEL -eq $TRIGGER_STACK_LEVEL ]; then
		for CMD in ${COMMANDS[@]}; do
			$CMD
		done
	fi
}
# Need this to remove log level when returning from a function
function trap_return_at_stack_level {
	# trap_return_at_stack_level <level> <code>
	trap_return_cmd_stack_add
}

function set_log_level {
	if [ ${LOG_LEVEL_VALUES[$1]:--1} -eq -1 ]; then
		log 'ERROR' "Cannot set log level to non-existent value '$1'."
		return 1
	fi
	LOG_LEVEL+=("$1")
}


#### EXAMPLE OF USE ####
set_log_level 'DEBUG'

declare -A TFILES=()
TFILES['STRIP_MARKDOWN']=$(mktemp --suffix "tmp_bash_example_strip_markdown")
TFILES['CELL_OUTPUTS']=$(mktemp --suffix "tmp_bash_example_cell_outputs")
TFILES['SCRIPT_WITH_OUTPUT']=$(mktemp --suffix "tmp_bash_example_script_with_output")

function cleanup {
	log 'INFO' "Signal recieved: ${@}"
	for TFILE in ${TFILES[@]}; do
		log 'ERROR' ${TFILE}
		if [ -f ${TFILE} ]; then
			rm -f ${TFILE}
		fi
	done
}

trap_func "cleanup" EXIT