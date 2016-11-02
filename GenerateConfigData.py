import ephem
import numpy as np



def Config(Date, **keyword_parameters):

    dtype = [('visitTime', np.float),('visitExpTime', np.float)]

    visitTime = 0; visitExpTime = 0

    if ('visitTime' in keyword_parameters):
        visitTime = keyword_parameters['visitTime']
    if ('visitExpTime' in keyword_parameters):
        visitExpTime = keyword_parameters['visitExpTime']

    Conf = np.array((visitTime, visitExpTime), dtype = dtype)

    np.save('NightDataInLIS/Config.npy', Conf)



Date = ephem.Date('2016/09/01 12:00:00.00') # times are in UT
Config(Date, visitTime = 30, visitExpTime = 30 )