"""
Contains support routines for holding data about "strategies" (i.e., different ways of doing things) that perform "tasks" (e.g., functions, operations, methods).
"""

from typing import Callable, Any, Self
import dataclasses as dc


@dc.dataclass(init=True, repr=True, order=False, eq=False, slots=True)
class TaskStratInfo:
	"""
	Base class that describes a strategy to peform a task
	"""
	name : str = dc.field(default=dc.MISSING, init=True, repr=True, hash=False, compare=False)
	description : str = dc.field(default=dc.MISSING, init=True, repr=True, hash=False, compare=False)
	callable : Callable[[...],Any] = dc.field(default=dc.MISSING, init=True, repr=False, hash=False, compare=False)


@dc.dataclass(init=True, repr=True, order=False, eq=False, slots=True)
class TaskStratSet:
	"""
	A set of strategies that perform a task, having a collection makes it easy to choose one from another
	"""
	description : str = dc.field(default=dc.MISSING, init=True, repr=True, hash=False, compare=False)
	_items : list[TaskStratInfo] = dc.field(default_factory=list, init=False, repr=True, hash=False, compare=False)
	_name_idx_map : dict[str,int] = dc.field(default_factory=dict, init=False, repr=True, hash=False, compare=False)

	def _get_item(self, key : int | str) -> TaskStratInfo:
		"""
		Return the strategy specified by `key`
		"""
		if type(key) is int:
			return self._items[key]
		else:
			return self._items[self._name_idx_map[key]]
		raise RuntimeError('This should never happen as key should be an int or a string in the self._name_idx_map dict')
	
	def __getitem__(self, key : int | str) -> TaskStratInfo:
		"""
		Return the strategy specified by `key`
		"""
		return self._get_item(key)
	
	def _set_item(self, value : TaskStratInfo) -> Self:
		"""
		Set save the passed strategy under the key `value.name`
		"""
		key : str = value.name
		idx = self._name_idx_map.get(key, len(self._items))
		self._name_idx_map[key] = idx
		self._items.append(value)
		return self
	
	def _del_item(self, key : int | str) -> Self:
		"""
		Delete the strategy specified by `key`
		"""
		if type(key) is int:
			del self._items[key]
			
			to_del_key = None
			for k,v in self._name_idx_map.items():
				if v == key:
					to_del_key = k
					break
			if to_del_key is not None:
				del self._name_idx_map[key]
			else:
				raise RuntimeError('This should never happen as `key` should always be an int or a string in `self._name_idx_map`')
		else:
			idx = self._name_idx_map.get(key, None)
			if idx is not None:
				del self._items[idx]
			else:
				raise RuntimeError('This should never happen as `key` should be in the `self._name_idx_map` dictionary')
			del self._name_idx_map[key]
		return self
	
	def add(self, *args : list[TaskStratInfo,...]) -> Self:
		"""
		Add a list of strategies to the set.
		"""
		for value in args:
			if value.name not in self._name_idx_map:
				self._set_item(value)
			else:
				raise RuntimeError('Could not add value with name "{value.name}" as an item of that name already exists in the StrategySet')
		return self
	
	def remove(self, key) -> Self:
		"""
		Remove a strategy specified by `key`
		"""
		if key in self._name_idx_map:
			self._del_item(key)
		# ignore key that is not present
		return self
	
	def get_callable(self, key : int | str) -> Callable[[...],Any]:
		"""
		Return the callable of the strategy specified by `key`
		"""
		return self.__getitem__(key).callable
	
	def get_description(self, key : int | str) -> Callable[[...],Any]:
		"""
		Return the description of the strategy specified by `key`
		"""
		return self.__getitem__(key).description
	
	def get_name(self, key : int | str) -> Callable[[...],Any]:
		"""
		Return the name of the strategy specified by `key`
		"""
		return self.__getitem__(key).name
	
	@property
	def names(self) -> tuple[str]:
		"""
		Return a tuple of the names of all the strategies in this set
		"""
		return tuple(item.name for item in self._items)
	
	@property
	def descriptions(self) -> tuple[str]:
		"""
		Return a tuple of the descriptions of all the strategies in this set
		"""
		return tuple(item.description for item in self._items)
	
	@property
	def callables(self) -> tuple[Callable[[...],Any]]:
		"""
		Return a tuple of the callables of all the strategies in this set
		"""
		return tuple(item.callable for item in self._items)
	
	def format_description(self, indent_level : int = 0, indent : str = '\t', with_default=True):
		"""
		Return a formatted string of the names and descriptions of the strategies in this set.
		"""
		lines = [indent*indent_level+self.description+f' (default : {self._items[0].name})' if with_default else '']
		for item in self._items:
			lines.append(indent*(indent_level+1)+item.name)
			lines.append(indent*(indent_level+2)+item.description)
		return '\n'.join(lines)
	
	def __call__(self, key, *args, **kwargs) -> Any:
		"""
		Call the strategy specified by 'key'
		"""
		return self.get_callable(key)(*args, **kwargs)
	
