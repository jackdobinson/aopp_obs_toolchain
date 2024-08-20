# Use strict mode
set -o errexit -o nounset -o pipefail
IFS=$'\n\t'

SCRIPT="$(readlink -f ${0})"
SCRIPT_DIR=${SCRIPT%/*}

SCRIPT_TO_BUILD="$1"
SCRIPT_AS_MD="${1%.*}.md"


# Run the script
bash ${SCRIPT_TO_BUILD}


# '/^#:begin\{HIDE\}/,/^#:end\{HIDE\}/d' -- delete everything between `#:begin{HIDE}` and `#:end{hide}` commands
# '/^#:HIDE/,+1p' -- delete `#:HIDE` and the line after that command
	
sed -E \
	-e '1 i\set -o errexit -o pipefail' \
	-e '/^#:begin\{HIDE\}/,/^#:end\{HIDE\}/d' \
	-e '/^#:HIDE/,+1d' \
	-e 's/^: << ---MD/: << ---MD\n/' \
	-e 's/^---MD/\n---MD/' \
	-e 's/^#:begin\{CELL\}/: << "---CELL"\n\`\`\`bash/' \
	-e 's/^#:end\{CELL\}/\`\`\`\n---CELL/' \
	-e "s/^: (<<.*$)/cat \1/" \
	${SCRIPT_TO_BUILD} | bash | cat > ${SCRIPT_AS_MD}
#-e "s/^: (<<.*\n)/cat \1 | ${SCRIPT_AS_MD}/" \
#
