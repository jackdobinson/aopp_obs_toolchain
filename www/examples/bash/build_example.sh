# Takes a specially formatted bash script and creates a markdown file
# that shows the commands run, the output, and explanatory text.
#
# # SYNTAX #
#
# A group of commands that are run together is called a 'cell', the start
# and end of a cell is denoted by special comments. 
#
# `#:begin{CELL}`
# : Begin a cell of commands
#
# `#:end{CELL}`
# : End the current cell of commands
#
# Everything between these comments will be run together and their output
# displayed below the commands themselves.
#
#
# To have commands within a cell that are not displayed in the final
# markdown document, there are also special comments. Commands that
# are hidden are still run, but are not displayed in the resulting markdown
# file.
#
# `#:begin{HIDE}`
# : Begin a group of commands hidden from the markdown output
#
# `#:end{HIDE}`
# : End a group of commands hidden from the markdown output
#
# `#:HIDE`
# : Hide the next line of commands from the markdown output
#
#
# Explanatory text is entered using **HEREDOC** formatting used as
# a way of creating block comments in the code. With the added bonus
# that we can decide to allow bash variable expansion within the
# explanatory text, or not as the situation requires.
#
# `: << ---MD`
# : Starts a HEREDOC that doesn't get sent anywhere, we use `---MD` to
#   denote that we expect the enclosed text to be *markdown*, but the
#   guard string could be anything as long as it does not appear in the
#   enclosed text. BASH parameters will be expanded within the text.
#
# `: << "---MD"`
# : Starts a HEREDOC in the same way as above, but BASH parameters are **not**
#   expanded. The text is used verbatim.
#
# `---MD`
# : Ends a HEREDOC that was opened with the guard string `---MD`,
#   whether the string was quoted or not.
#
#
# # PARSING #
#
# 
#


# Use strict mode
set -o errexit -o nounset -o pipefail
IFS=$'\n\t'

SCRIPT="$(readlink -f ${0})"
SCRIPT_DIR=${SCRIPT%/*}

SCRIPT_TO_BUILD=$(realpath $1)
SCRIPT_AS_MD="${1%.*}.md"



# Strip all explanatory text out of the script
sed -E -n \
	-e '0,/^: << ---MD/{/^: << ---MD/d;p}' \
	-e '/^---MD/,/^: << ---MD/{/^: << ---MD|^---MD/d;p}' \
	${SCRIPT_TO_BUILD} > test_markdown_stripped.sh




# Run each section of the script individually and store the results
# in a file.
bash <(sed -E -n \
-e "
1{
	i source <(cat << \"---CELL\"
	
}
s#^\s*SCRIPT=.*#SCRIPT=\"${SCRIPT_TO_BUILD}\"#g
s#\\$\{?BASH_SOURCE\}?#${SCRIPT_TO_BUILD}#g
s#\\$\{?0\}?#${SCRIPT_TO_BUILD}#g

/#:DUMMY/,+1d

p

/#:end\{CELL\}/{
	a ---CELL
	a )
	a echo '\#:DELIMITER_CELL_OUTPUT_END'
	a source <(cat << \"---CELL\"
	z
}

\${
	a ---CELL
	a )
	a echo '\#:DELIMITER_CELL_OUTPUT_END'
}
" test_markdown_stripped.sh) > test_output.sh



cat <(
I=0
while IFS= read -r LINE; do
	echo "$LINE"
	if [[ "$LINE" == "#:end{CELL}" ]]; then
		I=$((I+1))
		echo ': << ---RESULT'
		
		J=1
		while read -r RESULT; do
			if [[ "$RESULT" == "#:DELIMITER_CELL_OUTPUT_END" ]]; then
				J=$((J+1))
				if [[ $I == $((J-1)) ]]; then
					echo '---RESULT'
				fi
				continue
			fi
			if [[ $I == $J ]]; then
				echo "$RESULT"
			fi
		done < test_output.sh
	fi
	
done < ${SCRIPT_TO_BUILD}
) > test_run.sh


sed -E -n \
-e '
1 i set -o errexit -o pipefail

#######
# Remove everything that should be hidden
#######
/^#:begin\{HIDE\}/,/^#:end\{HIDE\}/d
/^#:HIDE/,+1d

#######
# If we are between result tags, stuff them  into the hold space
#######
/: << ---RESULT/,/^---RESULT/{
	x
	z
	x
	H
	b
}

#######
# If there is nothing between the results tags in the hold space,
# delete the hold space, don not bother showing empty results
#######
x
/^\n: << ---RESULT\n---RESULT$/d
x

#######
# Substitute tags so the markdown output makes sense when
# the HEREDOCs are run
#######
s/^---MD/\n---MD/
s/^#:begin\{CELL\}/: << "---CELL"\n\`\`\`bash/
s/^#:end\{CELL\}/\`\`\`\n\n---CELL/
s/^: << ---RESULT/: << ---RESULT\n\`\`\`bash/
s/^---RESULT/\`\`\`\n---RESULT/
s/^: (<<.*$)/cat \1/
p
' test_run.sh | bash | cat > ${SCRIPT_AS_MD}


# Remove temporary files
rm test_run.sh
rm test_output.sh
rm test_markdown_stripped.sh