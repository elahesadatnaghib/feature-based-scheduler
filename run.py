__author__ = 'Elahe'

import numpy as np
import ephem
import NightDataGenerator as DataGen
import FBDE
import time
import DBreadwrite as DB
import os



Site            = ephem.Observer()
Site.lon        = -1.2320792
Site.lat        = -0.517781017
Site.elevation  = 2650
Site.pressure   = 0.
Site.horizon    = 0.

preferences     = [1,1,10,2,3,10] # mission objective
# objective function
#objective = preferences[0] * cost_avg * -1 +
#            preferences[1] * slew_avg * -1 +
#            preferences[2] * alt_avg  *  1 +
#            preferences[3] * N_triple *  1 +
#            preferences[4] * N_double *  1 +
#            preferences[5] * N_single * -1

#F_weight : controller parameters
#F_weight        = np.array([ 1, 1, 1, 1, 1, 1, 1])  # all one
#F_weight        = np.array([2, 1, 1, 5, 3, 1, 2])   # educated guess
F_weight        = np.array([ 2.90846782,  2.15963323,  9.48473502,  7.74506438,  4.69452669,  5.33303562, 9.55935917]) * -1   # learning result

# F1: slew time cost 0~2
# F2: night urgency -1~1
# F3: overall urgency 0~1
# F4: altitude cost 0~1
# F5: hour angle cost 0~1
# F6: co-added depth cost 0~1
# F7: normalized brightness 0~1

s = time.time()

n_nights = 10 # number of the nights to be scheduled starting from 1st Sep. 2016


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



print('\n \nTotal elapsed time: {} sec'.format(time.time() - s))

