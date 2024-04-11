"""
Definitions of generic types so I can type hint more easily
"""

from typing import TypeVar, GenericAlias, NewType

T = TypeVar('T')

type NTuple[T,N] = GenericAlias(tuple, (T,)*N)

type NumVar = int | float

type ShapeVar[N] = NTuple[int,N]

