

class Pointer:
	"""
	Lets the user set and retrieve referenced variables like a pointer does.
	Not sure exactly how to emulate pointers to normal variables yet.

	USAGE
	a_ptr = Pointer(7)
	b_ptr = Pointer(8)

	a_ptr.ref = b_ptr.ref
	print(a_ptr) # '8'

	b_ptr.val = 10
	print(a_ptr) # '10'
	"""
	__slots__=("__ref",)
	def __init__(self, val):
		self.ref = [val]
	@property
	def ref(self):
		""" Get the "reference" object, i.e. the internal list"""
		return(self.__ref)
	@ref.setter
	def ref(self, ref_list):
		""" Set the reference object, i.e. change the internal list"""
		self.__ref = ref_list
	@property
	def val(self):
		""" get the value stored in the reference, i.e. the single list element"""
		return(self.__ref[0])
	@val.setter
	def val(self, val):
		""" set the value stored in the referece, i.e. the single list element"""
		self.__ref[0] = val
