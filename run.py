__author__ = 'Elahe'

import numpy as np
import ephem
import NightDataGenerator as DataGen
import FBDE
import time
import DBreadwrite as DB
import Graphics as GP
import os



Site            = ephem.Observer()
Site.lon        = -1.2320792
Site.lat        = -0.517781017
Site.elevation  = 2650
Site.pressure   = 0.
Site.horizon    = 0.

preferences     = [1,1,4,0,3,5] # mission objective
# objective function
#objective = preferences[0] * average cost * -1 +
#            preferences[1] * average slew time * -1 +
#            preferences[2] * average altitude  *  1 +
#            preferences[3] * No. of triple visits *  1 +
#            preferences[4] * No. of double visits *  1 +
#            preferences[5] * No. of single visits * -1

#F_weight : controller parameters
#F_weight        = np.array([ 1, 1, 1, 1, 1, 1, 1])  # all one
#F_weight        = np.array([2, 1, 1, 5, 3, 1, 2])  # educated guess
#F_weight        = np.array([ 2.90846782,  2.15963323,  9.48473502,  7.74506438,  4.69452669,  5.33303562, 9.55935917])    # learning result
F_weight        = [ 1.29964032,  9.83017599,  5.21240644,  6.3694487,   0.15822261,  7.11310888, 8.74563025]               # learning result

# F1: slew time cost 0~2
# F2: night urgency -1~1
# F3: overall urgency 0~1
# F4: altitude cost 0~1
# F5: hour angle cost 0~1
# F6: co-added depth cost 0~1
# F7: normalized brightness 0~1

# immediate reward reward = F_weight[0] * F1 + F_weight[1] * F2 + F_weight[2] * F3 + F_weight[3] * F4 + F_weight[4] * F5 + F_weight[5] * F6 + F_weight[6] * F7
s = time.time()

n_nights = 3 # number of the nights to be scheduled starting from 1st Sep. 2016


# Delete previous database
try:
    os.remove('FBDE.db')
except:
    pass

for i in range(1, n_nights+1):
    Date = ephem.Date('2016/09/{} 12:00:00.00'.format(i)) # times are in UT
    # Delete previous history dependent data
    try:
        os.remove('NightDataInLIS/t_last_visit{}.lis'.format(int(ephem.julian_date(Date))))
        os.remove('NightDataInLIS/tot_N_visit{}.lis'.format(int(ephem.julian_date(Date))))
    except:
        pass

    # generate data of the night (both history dependent and independent)
    DataGen.Night_data(Date,Site)
    t1 = time.time()
    print('\nData generated in {} sec'.format(t1 - s))

    # create scheduler
    scheduler = FBDE.Scheduler(Date, Site, F_weight, preferences)
    t2 = time.time()
    print('\nData imported in {} sec'.format(t2 - t1))

    # schedule
    scheduler.schedule()
    t3 = time.time()
    print('\nScheduling finished in {} sec'.format(t3 - t2))

    # write on DB
    DB.DBreadNwrite('w', Date)
    t4 = time.time()
    print('\nWriting on DB finished in {} sec'.format(t4 - t3))

    # graphics

    # Animation specifications
    FPS = 10            # Frame per second
    Steps = 100          # Simulation steps
    MP4_quality = 300   # MP4 size and quality

    PlotID = 2        # 1 for one Plot, 2 for including covering pattern
    GP.Visualize(Date, PlotID ,FPS, Steps, MP4_quality, 'Visualizations/LSST1plot{}.mp4'.format(int(ephem.julian_date(Date))), showClouds= False)

print('\n \nTotal elapsed time: {} sec'.format(time.time() - s))

