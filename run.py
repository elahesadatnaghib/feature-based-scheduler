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

preferences     = [1, 1, 4, 0, 3, 10] # mission objective
# objective function
#objective = preferences[0] * average cost * -1 +
#            preferences[1] * average slew time * -1 +
#            preferences[2] * average altitude  *  1 +
#            preferences[3] * No. of triple visits *  1 +
#            preferences[4] * No. of double visits *  1 +
#            preferences[5] * No. of single visits * -1

#F_weight : controller parameters
#F_weight        = np.ones([8,7])
F_weight        = np.reshape(np.array([ 4.20719114,  7.30495229,  3.22797504,  7.36876567,  2.48813029,  2.25788253,
                                        2.06713858,  0.6136642,   4.13515909,  1.91470662,  8.47890309,  4.68662087,
                                        8.29988236,  4.70413934,  2.78585885,  6.60923852,  3.94503914,  5.62561784,
                                        4.01397898,  5.59309747,  4.00137767,  7.85457785,  7.9947913,   2.00940157,
                                        6.40087058,  5.27213064,  5.41139189,  5.7581916,   6.52381396,  5.79545407,
                                        6.7566571,   4.81852891,  4.79779777,  4.4556204,   1.43700385,  6.70733977,
                                        2.47125883,  4.53605309,  5.26014559,  8.0306582,   4.56409016,  4.13744006,
                                        1.96368802,  4.61564592,  4.25298633,  3.56414258,  4.92874663,  5.87943215,
                                        9.32594932,  1.39163788,  9.16930444,  3.62697418,  3.18903897,  4.23514478,
                                        7.15683848,  6.20172087]),[8,7])  # learning result

# F1: slew time cost 0~2
# F2: night urgency -1~1
# F3: overall urgency 0~1
# F4: altitude cost 0~1
# F5: hour angle cost 0~1
# F6: co-added depth cost 0~1
# F7: normalized brightness 0~1

# immediate reward reward = F_weight[0] * F1 + F_weight[1] * F2 + F_weight[2] * F3 + F_weight[3] * F4 + F_weight[4] * F5 + F_weight[5] * F6 + F_weight[6] * F7
s = time.time()

n_nights = 1 # number of the nights to be scheduled starting from 1st Sep. 2016


# Delete previous database
try:
    os.remove('FBDE.db')
except:
    pass

for i in range (n_nights):
    Date = ephem.Date('2015/07/1 12:00:00.00') + i # times are in UT
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
    scheduler = FBDE.Scheduler(Date, Site, F_weight, preferences, micro_train = False)
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

    # Animation specifications
    FPS = 10            # Frame per second
    Steps = 300          # Simulation steps
    MP4_quality = 300   # MP4 size and quality

    PlotID = 1        # 1 for one Plot, 2 for including covering pattern
    GP.Visualize(Date, PlotID ,FPS, Steps, MP4_quality, 'Visualizations/LSSTplot{}.mp4'.format(int(ephem.julian_date(Date))), showClouds= True)

print('\n \nTotal elapsed time: {} sec'.format(time.time() - s))

