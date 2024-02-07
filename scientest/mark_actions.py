import scientest.exceptions


def action_skip(payload):
	raise scientest.exceptionsons.TestSkippedException(payload)

action_map = {
	'skip' : action_skip

}


