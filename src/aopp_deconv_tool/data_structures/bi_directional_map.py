
class BiDirectionalMap:
	"""
	A map that can efficiently go from A->B and B->A.
	"""
	def __init__(self, seed_dict={}):
		self.is_forward = True
		self.forward_dict = dict((k,v) for k,v in seed_dict.items())
		self.backward_dict = dict((v,k) for k,v in seed_dict.items())
	
	def __getattr__(self, name):
		proxy = self.forward_dict if self.is_forward else self.backward_dict
		if hasattr(proxy, name):
			return getattr(proxy, name)
		else:
			raise AttributeError(f'No attribute {name} on {type(self).__name__}')
	
	@property
	def backward(self):
		self.is_forward = False
		return self
	
	def __getitem__(self, key):
		if self.is_forward:
			return self.forward_dict[key]
		else:
			self.is_forward = True
			return self.backward_dict[key]
	
	def __contains__(self, key):
		if self.is_forward:
			return key in self.forward_dict
		else:
			self.is_forward = True
			return key in self.backward_dict
	
	def __setitem__(self, key, value):
		if self.is_forward:
			self.forward_dict.__setitem__(key,value)
			self.backward_dict.__setitem__(value,key)
		else:
			self.is_forward = True
			self.forward_dict.__setitem__(value,key)
			self.backward_dict.__setitem__(key,value)
		return
	
	def update(self, *args, **kwargs):
		is_forward_flag = self.is_forward
		for k,v in dict(*args, **kwargs).items():
			self.is_forward = is_forward_flag
			self.__setitem__(k,v)
	
	def __repr__(self):
		return f'{type(self).__name__}({str(self.forward_dict)})'
