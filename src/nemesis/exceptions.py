#!/usr/bin/env python3
import sys, os

class NemesisError(Exception):
	"""Base class for errors with nemesis module"""
	pass

class NemesisReadError(NemesisError):
	"""Raised when a file is read incorrectly/has a different structure than what we assume"""
	pass
