#!/usr/bin/env python3
import sys, os

class FitscubeException(Exception):
	"""Base class for exceptions within fitscube module"""
	pass

class FitscubeNotImplementedError(FitscubeException):
	"""Raised when a part of the code has not been implemented yet"""
	pass
