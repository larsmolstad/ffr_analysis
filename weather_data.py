""" Downloads weather data from yr.
Usage:
data = weather_data.data
data.update()# downloads new data from yr
data.get_precip(time.time()-86400) # for example
data.get_temp(time.mktime((2017,12,24,20,30,0,0,0,0))) # temperature christmas eve 2017 at 8:30pm
"""
import os
import pickle
import numpy as np
import pandas as pd
import resdir
import weather_data_from_yr


class Weather_data(object):

    def __init__(self, do_update=False):
        if do_update:
            self.update()
        else:
            self.make_assignments()

    def update(self):
        weather_data_from_yr.update_weather_data()
        
    def make_assignments(self):
        self.data = pickle.load(open(weather_data_from_yr.DATA_FILE_NAME))
        x = [np.array(x) for x in self.get_temps_and_precip(self.data)]
        self.all_times, self.all_temps, self.all_precips = x

        temp_nans = np.isnan(self.all_temps)
        precip_nans = np.isnan(self.all_precips)

        self.temp_times = self.all_times[~temp_nans]
        self.temperature = self.all_temps[~temp_nans]

        # I just have to assume the precipitation measurement was alway working
        # so nan means zero
        self.precip_times = self.all_times
        self.precip = self.all_precips[:]
        self.precip[precip_nans] = 0

    def get_temps_and_precip(self, q):
        def choose_T(T):
            if T[0]:
                return T[0]  # measured
            if T[1] and T[2]:
                return (T[1] + T[2]) / 2
            if T[1]:
                return T[1]
            if T[2]:
                return T[2]
            return np.nan
        t = [x[0] for x in q]
        T = [choose_T(x[1][1]) for x in q]
        precip = [(x[1][2] if x[1][2] else np.nan) for x in q]
        return t, T, precip

    def get_temp(self, t):
        return np.interp(t, self.temp_times, self.temperature)

    def get_precip(self, t):
        return np.interp(t, self.precip_times, self.precip)

    def get_precip2(self, t, interval=[-86400, 0]):
        """gets the average precipitation over the time interval around t"""
        p = np.zeros(len(t))
        # hmmm
        for i, ti in enumerate(t):
            I0 = np.searchsorted(self.precip_times, ti + interval[0])
            I1 = np.searchsorted(self.precip_times, ti + interval[1])
            if I0 == I1:
                p[i] = self.precip[I0]
            else:
                p[i] = np.mean(self.precip[I0:I1])
        return p

data = Weather_data()
