#!/usr/bin/env python3

import webservices.observatory_codes
import astropy as ap
from astroquery.jplhorizons import Horizons

# Provides object IDs for JPL-Horizons ephemeridies system
# Update as needed
object_ids = {
		'neptune':	(899, 'majorbody'),
		# Moons of neptune #
			'triton':		(801, 'majorbody'),
			'nereid':		(802, 'majorbody'),
			'naiad':		(803, 'majorbody'),
			'thalassa':		(804, 'majorbody'),
			'despina':		(805, 'majorbody'),
			'galatea':		(806, 'majorbody'),
			'larissa':		(807, 'majorbody'),
			'proteus':		(808, 'majorbody'),
			'halimede':		(809, 'majorbody'),
			'psamathe':		(810, 'majorbody'),
			'sao':			(811, 'majorbody'),
			'laomedeia':	(812, 'majorbody'),
			'neso':			(813, 'majorbody'),
			'2004n1':		(814, 'majorbody')
		# ----- -- ------- #
	}

observatory_codes_dataset = webservices.observatory_codes.MinorPlanetCenter()



def query_eph_to_ids(obj, origin_observatory, observation_date, fields, ids=None):
	"""
	Returns a dictionary of fields selected from the jplhorizons database and renamed using list 'ids'
	""" 
	jpl_obj = query_object_at_instant(origin_observatory, obj, observation_date)
	jpl_eph = jpl_obj.ephemerides()	
	adict = {}
	if ids is not None:
		for f, i in zip(fields, ids):
			adict[i] = jpl_eph[f][0]
	else:
		for f in fields:
			adict[f] = jpl_eph[f][0]
	return(adict)

def query_object_at_instant(observer_location, target, date):
	tgt = object_ids[target]

	if type(observer_location) is dict:
		loc = observer_location
	elif observer_location in observatory_codes_dataset:
		loc = observer_location
	else:
		loc = observatory_codes_dataset.get_observatory_data_from_name(observer_location).id_code
	t1 = ap.time.Time(date, format='fits')
	#print(t1.mjd)
	print(tgt[0], loc, [t1.jd], tgt[1])
	return(Horizons(id=tgt[0], location=loc, epochs=[t1.jd], id_type=tgt[1]))


if __name__=='__main__':
	print(query_eph_to_ids('neptune', 'eso-paranal', '2011-12-05T06:55:32.4353', ('r', 'PDObsLat', 'delta', 'delta_rate', 'ang_width')))
