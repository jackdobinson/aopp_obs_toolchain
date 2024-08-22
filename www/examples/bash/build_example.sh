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

declare -A TFILES=()
TFILES['STRIP_MARKDOWN']=$(mktemp --suffix "tmp_bash_example_strip_markdown")
TFILES['CELL_OUTPUTS']=$(mktemp --suffix "tmp_bash_example_cell_outputs")
TFILES['SCRIPT_WITH_OUTPUT']=$(mktemp --suffix "tmp_bash_example_script_with_output")

function cleanup {
	for TFILE in ${TFILES[@]}; do
		if [ -f ${TFILE} ]; then
			echo "Removing temp file: $TFILE"
			rm -f ${TFILE}
		fi
	done
}

trap "cleanup" EXIT

# Only keep things that are not the explanatory text
sed -E -n \
-e '
#######
# Print everything until the start of the first
# explanatory text section.
#######
0,/^: << ---MD/{
	/^: << ---MD/d
	p
}

#######
# Print everything between explanatory text sections
#######
/^---MD/,/^: << ---MD/{
	/^: << ---MD|^---MD/d
	p
}
' ${SCRIPT_TO_BUILD} > ${TFILES['STRIP_MARKDOWN']}

echo "Markdown Stripped"

#
# NOTE: COMMANDS ARE EXECUTED HERE
#
# Run each section of the script individually and store the results
# in a file. Use `source` command to execute each code cell in the same
# bash shell, and use HEREDOCs combine with process substitution to 
# execute commands in discrete chunks.
bash <(sed -E -n \
-e '
#######
# Execute all the code from the top, start
# process substitution and HEREDOC.
#######
1{
	i source <(cat << "---CELL"
}

#######
# Because the bash shell we are redirecting
# the output to will not have any arguments,
# or a file to populate ${BASH SOURCE} with,
# replace the $0 argument and ${BASH_SOURCE}
# with the path to the script we are
# building, as if it was being executed
# normally.
#######
' -e "
s#^\s*SCRIPT=.*#SCRIPT=\"${SCRIPT_TO_BUILD}\"#g
s#\\$\{?BASH_SOURCE\}?#${SCRIPT_TO_BUILD}#g
s#\\$\{?0\}?#${SCRIPT_TO_BUILD}#g

" -e '

#######
# Remove any commands marked as DUMMY commands
#######
/#:DUMMY/,+1d

#######
# Print each line of the command-only script
#######
p

#######
# When we get to the end of a cell, close the
# HEREDOC and process substitution, print a
# line to tell us where the output of the
# current code cell ends, and start a new
# code cell by starting process substitution
# and a HEREDOC.
#######
/#:end\{CELL\}/{
	a ---CELL
	a )
	a echo "#:DELIMITER_CELL_OUTPUT_END"
	a source <(cat << "---CELL"
	z
}

#######
# At the end of the bash command input, close
# the HEREDOC and process substitution, print
# a line to tell us where the output of the
# last code cell ends.
#######
${
	a ---CELL
	a )
	a echo "#:DELIMITER_CELL_OUTPUT_END"
}
' ${TFILES['STRIP_MARKDOWN']}) > ${TFILES['CELL_OUTPUTS']}

echo "Commands executed and outputs recorded"


# Assemble the combined text (commands + command results + explanatory text)
# into a single file using the results we created by running the commands
# in the above step.
cat <(
I=0
while IFS= read -r LINE; do
	echo "$LINE"
	
	# When we are at the end of a cell,
	# copy the results from the output file
	if [[ "$LINE" == "#:end{CELL}" ]]; then
		# Keep track of which cell we have just
		# passed. We want to insert the results
		# for the correct cell.
		I=$((I+1))
		
		# Start the result HEREDOC
		echo ': << ---RESULT'
		
		# Read through the output of the command cells
		# The first group of outputs is the result of the
		# first command cell
		J=1
		while read -r RESULT; do
			# When we get to the end of a cell output
			if [[ "$RESULT" == "#:DELIMITER_CELL_OUTPUT_END" ]]; then
				# Incrememnt the cell output counter
				J=$((J+1))
				
				# If the group of outputs we just exited
				# are the outputs of the current cell
				# end the result HEREDOC
				if [[ $I == $((J-1)) ]]; then
					# end the result HEREDOC
					echo '---RESULT'
				fi
				# Do not print the cell output end delimiter
				continue
			fi
			
			# If we are in the results group for the
			# current cell, print each line of the results.
			if [[ $I == $J ]]; then
				echo "$RESULT"
			fi
		done < ${TFILES['CELL_OUTPUTS']}
	fi
	
done < ${SCRIPT_TO_BUILD}
) > ${TFILES['SCRIPT_WITH_OUTPUT']}

echo "Combined (commands + results + markdown) file assembled"


# Go through the combined (commands + command results + explanatory text) file.
# Change cell begin/end commands to verbatim HEREDOCs, remove empty RESULT
# HEREDOCs, make RESULT HEREDOCs verbatim, and wrap RESULT block contents 
# in markdown code fences.
# Also ensure enough new lines between CODE, MARKDOWN, RESULT sections.
#
# The output of the SED command is run in a bash shell and the output
# is a markdown file that documents the originally passed script.
sed -E -n \
-e '
#######
# Turn on "safe mode" for assembling the markdown output
#######
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
# delete the hold space, do not bother showing empty results.
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
s/^: << ---RESULT/: << "---RESULT"\n\`\`\`bash/
s/^---RESULT/\`\`\`\n---RESULT/
s/^: (<<.*$)/cat \1/
p
' ${TFILES['SCRIPT_WITH_OUTPUT']} | bash | cat > ${SCRIPT_AS_MD}

echo "Markdown documentation of script written to \"${SCRIPT_AS_MD}\""