""" Downloads weather data from yr.
Usage:
data = weather_data.data
data.update()# downloads new data from yr
data.get_precip(time.time()-86400) # for example
data.get_temp(time.mktime((2017,12,24,20,30,0,0,0,0))) # temperature christmas eve 2017 at 8:30pm
"""
import pickle
import numpy as np
import weather_data_from_metno


class Weather_data(object):

    def __init__(self, do_update=False):
        if do_update:
            self.update()
        else:
            self.make_assignments()

    def update(self):
        weather_data_from_metno.update_weather_data()

    def make_assignments(self):
        def get(dct, key, default):
            if key in dct:
                return dct[key]
            else:
                return default
        try:
            self.data = weather_data_from_metno.get_stored_data()
        except FileNotFoundError:
            self.data = []
        self.all_times = np.array([x[0] for x in self.data])
        self.all_temps = np.array([get(x[1], 'air_temperature', np.nan) for x in self.data])
        self.all_precips = np.array([get(x[1], 'sum(precipitation_amount PT1H)', np.nan) for x in self.data])

        temp_nans = np.isnan(self.all_temps)
        precip_nans = np.isnan(self.all_precips)

        self.temp_times = self.all_times[~temp_nans]
        self.temperature = self.all_temps[~temp_nans]

        # I just have to assume the precipitation measurement was alway working
        # so nan means zero
        self.precip_times = self.all_times
        self.precip = self.all_precips[:]
        self.precip[precip_nans] = 0

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

# def weatherdatatest(day=(2015,2,10)):
#     import time, webbrowser
#     while len(day) < 9:
#         day = (*day, 0)
#     day = list(day)
#     for i in range(1,25):
#         day[3] = i
#         t = time.mktime(tuple(day))
#         print(time.ctime(t), "    {:.2f}".format(data.get_temp(t)))
#     date_string = "{}-{}-{}".format(*day[:3])
#     webbrowser.open(weather_data_from_yr.make_url(date_string), new=0)
#     return time.ctime(t)

