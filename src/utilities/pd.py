#!/usr/bin/env python3
"""
Contains helper functions for pandas
"""

import pandas as pd
import numpy as np
import datetime

def df_like(df, d=None):
	# assume df is in the same format as d
	if d is None:
		d = np.empty(0, dtype=[(k,df[k].dtype) for k in df.columns])
	data = {}
	for i, k in enumerate(df.columns):
		data[k] = np.array(d[:,i].ravel(), dtype=df[k].dtype)
	return(pd.DataFrame(data))


def split_on_events(
		df,
		col,
		events,
		offset_from_event = True
	):
	"""
	Split dataframe on intervals between 'events' depending on value of 'col'
	"""
	event_pair = tuple(zip(events[:-1],events[1:]))
	for e1, e2 in event_pair:
		e_slice = df.values[(e1<=df[col]) & (df[col]<e2),:]
		sdf = df_like(df, e_slice)
		if type(offset_from_event) is bool and offset_from_event:
			sdf[col] -= e1
		elif type(offset_from_event) is str:
			sdf[offset_from_event] = sdf[col]-e1
		yield(sdf)


def interp_to_time_grid(
		df,
		col,
		dt = datetime.timedelta(seconds=10),
		t_unit = datetime.timedelta(seconds=1),
		t_zero = datetime.datetime(2021,1,1),
		t_lims = None,
		asfloat=True,
		interp_empty_flag=True
	):
	
	def to_correct_np_time(x):
		if type(x) in (datetime.datetime, np.datetime64):
			x = np.datetime64(x)
		elif type(x) in (datetime.timedelta, np.timedelta64):
			x = np.timedelta64(x)
		else:
			raise TypeError(f'Could not convert value {x} to numpy "np.datetime64" or "np.timedelta64" object')
		return(x)
	
	def np_time_to_float(x, t_zero=np.datetime64(), t_unit=np.timedelta64(1,'s')):
		if type(x) is np.datetime64:
			return((x-t_zero)/t_unit)
		elif type(x) is np.timedelta64:
			return(x/t_unit)
		raise TypeError(f'Could not convert numpy time object {x} to float')
		
		
	#df_like(df, d=None) # DEBUGGING
	# if no data, return copy of dataframe
	#if df[col].size < 1:
	#	gdf = pd.DataFrame(data=None, columns=df.columns, dtype=float) if asfloat else df.copy()
	#	return(gdf)
	is_empty_flag = True if df.size==0 else False
	if is_empty_flag and not interp_empty_flag:
		gdf = pd.DataFrame(data=None, columns=df.columns, dtype=float) if asfloat else df.copy()
		return(gdf)
	
	# get everything to correct time/type
	t_unit = to_correct_np_time(t_unit)
	t_zero = to_correct_np_time(t_zero)	
	dx = np.timedelta64(dt)/t_unit
	
	# get time axis of dataframe as a float
	t = (df[col] - t_zero)/t_unit
	
	# calculate limits of time-axis grid
	if t_lims is None:
		t_min = np.min(t)
		t_max = np.max(t)
	else:
		t_min, t_max = [np_time_to_float(to_correct_np_time(_x), t_zero=t_zero, t_unit=t_unit) for _x in t_lims]
	
	# get time axis grid as float
	t_grid = np.arange(t_min, t_max, dx)
	
	# construct new dataframe from interpolations along time-axis grid
	gdf = pd.DataFrame()
	gdf[col] = t_grid if asfloat else t_grid*t_unit + t_zero
	for k in df.columns:
		if k == col:
			continue
		if is_empty_flag:
			gdf[k] = np.interp(t_grid, np.array([0]), np.array([np.nan]), left=np.nan, right=np.nan)
		else:
			gdf[k] = np.interp(t_grid, t, df[k].values, left=np.nan, right=np.nan)
			
	return(gdf)




