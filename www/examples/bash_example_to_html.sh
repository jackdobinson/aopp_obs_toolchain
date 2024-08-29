#!/usr/bin/env bash
#
# Takes a specially formatted bash script and creates a markdown file
# that shows the commands run, the output, and explanatory text.
#
# # EXAMPLE #
#
# ```
# $ bash build_example.sh example_1.sh
# ```
#
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
# `#:DUMMY`
# : Display, but do not run, the next line of commands
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

LOG_DIR="${SCRIPT_DIR}/logs/bash_example_to_html"
mkdir -p ${LOG_DIR}

SCRIPT_TO_BUILD=$(realpath $1)
SCRIPT_AS_MD="${SCRIPT_TO_BUILD%.*}.md"


echo "Processing: ${SCRIPT_TO_BUILD}"
echo "Producing: ${SCRIPT_AS_MD}."
if [ ${SCRIPT_TO_BUILD} -nt ${SCRIPT_AS_MD} ]; then
	echo "Process file is newer than product file."
else
	echo "Process file is older than product file."
	echo "Not rebuilding product."
	echo "Exiting..."
	exit 0
fi

echo "Rebuilding product..."

# Ensure we load up the correct virtual environment
REPO_DIR="${HOME}/Documents/repos/aopp_obs_toolchain"
source ${REPO_DIR}/.venv_3.12.2/bin/activate

declare -A TFILES=()
TFILES['STRIP_MARKDOWN']=$(mktemp "tmp_bash_example_strip_markdown.XXXXXX")
TFILES['CELL_OUTPUTS']=$(mktemp "tmp_bash_example_cell_outputs.XXXXXX")
TFILES['SCRIPT_WITH_OUTPUT']=$(mktemp "tmp_bash_example_script_with_output.XXXXXX")

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
# Remove any commands marked as DUMMY commands
#######
/#:DUMMY/,+1d

#######
# Print everything between explanatory text sections
#######
/^---MD/,/^: << ---MD/{
	/^: << ---MD|^---MD/d
	p
}
' ${SCRIPT_TO_BUILD} > ${TFILES['STRIP_MARKDOWN']}

echo "Markdown Stripped"

#DEBUGGING
cp ${TFILES['STRIP_MARKDOWN']} ${LOG_DIR}/script_without_markdown.sh


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
' ${TFILES['STRIP_MARKDOWN']}) | cat > ${TFILES['CELL_OUTPUTS']}

echo "Commands executed and outputs recorded"

# DEBUGGING
cp ${TFILES['CELL_OUTPUTS']} ${LOG_DIR}/cell_output.txt


# Assemble the combined text (commands + command results + explanatory text)
# into a single file using the results we created by running the commands
# in the above step.
cat <(
I=0
while IFS= read -r LINE || [[ -n "$LINE" ]]; do
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

# DEBUGGING
cp ${TFILES['SCRIPT_WITH_OUTPUT']} ${LOG_DIR}/script_with_output.txt

# Go through the combined (commands + command results + explanatory text) file.
# Change cell begin/end commands to verbatim HEREDOCs, remove empty RESULT
# HEREDOCs, make RESULT HEREDOCs verbatim, and wrap RESULT block contents 
# in markdown code fences.
# Also ensure enough new lines between CODE, MARKDOWN, RESULT sections.
#
# The output of the SED command is run in a bash shell and the output
# is a markdown file that documents the originally passed script.
cat ${TFILES['SCRIPT_WITH_OUTPUT']} \
| sed -E -n \
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
/^#:DUMMY/d

#######
# If we are between result tags, stuff them  into the hold space
#######
/: << ---RESULT/,/^---RESULT/{
	# If we are at the start of result tag nuke hold space
	/^: << ---RESULT/{
		x
		z
		x
	}
	# append pattern to hold space
	H
	# If we are at the end of the result tag
	/^---RESULT$/{
		# pull stored pattern out of hold space
		x
		#l
		# If we only have the start and end tags, remove it and restart cycle
		/^\n: << ---RESULT\n---RESULT$/{
			d
		}
		# otherwise print the tag and restart cycle
		p
		b
	}
	# if we are not at the end, restart the cycle
	b
}
#######
# If we are in a cell, remove leading and trailing whitespace, condence repeated newlines into one newline
#######
/^#:begin\{CELL\}/,/^#:end\{CELL\}/{
	# At the start of a cell, push cell into holdspace
	/^#:begin\{CELL\}/{
		h
		b
	}
	
	# If the pattern is an empty line
	/^[[:space:]]*$/{
		# delete pattern
		z
		# If the last thing stored in hold space is a newline, do nothing
		x
		/\n$/{x;b}
		x
		# otherwise, append pattern to hold space
		H
		# move to next line
		b
	}
	# otherwise, append to holdspace
	H
	# If we are at the end of a cell
	/^#:end\{CELL\}/{
		# If the last thing in holdspace is whitespace followed by a newline, replace it with a single newline
		x
		s/[[:space:]]*\n#:end\{CELL\}$/\n#:end\{CELL\}/
		x
		# print holdspace
		x
		p
		# reset holdspace
		z
		x
		# move to next line
		b
	}
	# otherwise move to next line
	b
}
p
' \
| sed -E -n \
-e '
#######
# If there is nothing between the results tags in the hold space,
# delete the hold space, do not bother showing empty results.
#######
#x
#/^\n: << ---RESULT\n---RESULT$/d
#x

: PROCESS_HEREDOCS

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
' | bash | cat > ${SCRIPT_AS_MD}

echo "Markdown documentation of script written to \"${SCRIPT_AS_MD}\""