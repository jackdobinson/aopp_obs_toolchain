#!/usr/bin/env python3

planet_physical_parameters = dict(
	neptune = dict(
		mean_radius = 24722,
		mean_radius_err = 19,
		mean_radius_unit = 'km',
		equatorial_radius = 24764,
		equatorial_radius_err = 15,
		equatorial_radius_unit = 'km',
		polar_radius = 24341,
		polar_radius_err = 30,
		polar_radius_unit = 'km',
		mass = 17.15,
		mass_err = 0.01,
		mass_unit = 'M_earth',
	)
)


if __name__=='__main__':
	for planet, data in planet_physical_parameters.items():
		print(planet)
		for k, v in data.items():
			print(f'\t{k}\n\t\t{v}')














