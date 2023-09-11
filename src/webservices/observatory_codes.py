#!/usr/bin/env python3

import dataclasses as dc
import urllib.request
import numpy as np
import re
from collections import namedtuple


ObservatoryData = namedtuple('ObservatoryData', 'id_code, position, name')

@dc.dataclass
class GeocentricCoordinates:
	"""
	See <https://space.stackexchange.com/questions/35623/how-were-the-geodetic-and-geocentric-latitudes-of-the-space-shuttle-defined-and/35654#35654>
	section about geocentric coordinates.
	"""
	longitude : float # degrees
	pc_cos : float # parallax constant for cosine term
	pc_sin : float # parallax constant for sine term

	@classmethod
	def from_long_lat_rad(cls, longitude : float, latitude : float, radial_distance : float, radians=False):
		latitude = np.pi*latitude/180 if not radians else latitude
		pc_cos = radial_distance*np.cos(latitude)
		pc_sin = radial_distance*np.sin(latitude)
		return(cls(longitude if not radians else (180.0/np.pi)*longitude, pc_cos, pc_sin))

	def as_ECEF(self):
		"""
		coverting to earth-centered, earth-fixed cartesian coordinates
		"""
		return(np.cos(self.longitude)*self.pc_cos, np.sin(self.longitude)*self.pc_cos, self.pc_sin)
	
def try_cast(cast, value, fail_cast_value=None):
	try:
		return(cast(value))
	except:
		return(fail_cast_value)


@dc.dataclass
class MinorPlanetCenter:
	url = "https://minorplanetcenter.net/iau/lists/ObsCodes.html"
	uf_table = None
	url_encoding = 'utf-8'

	alternate_names = {
		'Cerro Paranal' : ('eso-paranal',),
	}

	def __post_init__(self):
		if self.uf_table is None:
			with urllib.request.urlopen(self.url) as f:
				self.uf_table = f.read()[6:-7].decode(self.url_encoding)

			self.entries = []
			for i, aline in enumerate(self.uf_table.split('\n')):
				if i == 0: 	continue # skip column headers
				_id_code = aline[:3]
				_position = GeocentricCoordinates(try_cast(float, aline[4:13]), try_cast(float, aline[13:21]), try_cast(float, aline[21:30]))
				_name = aline[30:]
				self.entries.append(ObservatoryData(_id_code, _position, _name))

			self.id_code2idx = dict((obs_data.id_code, i) for i, obs_data in enumerate(self.entries))
		return

	def __getitem__(self, key):
		if key not in self.id_code2idx:
			raise ValueError(f'Observatory identifier code {key} not in dataset...')
		return(self.entries[self.id_code2idx[key]])

	def __contains__(self, key):
		return(key in self.id_code2idx)

	def get_observatory_data_from_name(self, name, use_regex=True):
		if use_regex:
			name = re.compile(name)
			is_match = lambda a, b: a.match(b)
		else:
			is_match = lambda a, b: a==b
		for obs_data in self.entries:
			if is_match(name, obs_data.name) or any(is_match(name, x) for x in self.alternate_names.get(obs_data.name, tuple())):
				return(obs_data)
		raise ValueError(f'No entries with name matching "{name}" found in dataset...')
			



if __name__=='__main__':
	mpc = MinorPlanetCenter()
	print(mpc['309']) #look up via id_code
	print(mpc.get_observatory_data_from_name(r'(?i:.*paranal)')) # lookup via regex
	print(mpc.get_observatory_data_from_name('eso-paranal', use_regex=False)) # lookup via alternate name
