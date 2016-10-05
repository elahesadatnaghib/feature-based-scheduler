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
#objective = preferences[0] * average cost * -1 +
#            preferences[1] * average slew time * -1 +
#            preferences[2] * average altitude  *  1 +
#            preferences[3] * No. of triple visits *  1 +
#            preferences[4] * No. of double visits *  1 +
#            preferences[5] * No. of single visits * -1

#F_weight : controller parameters
#F_weight        = np.ones([8,7])
F_weight        = np.reshape(np.array([3.82825499,  5.0246024,   0.39586326,  5.22392334,  5.69871546,  1.07005201,
                            2.80493447,  9.76303104,  6.21157402,  3.26728548,  5.2239269,   1.61210018,
                            1.90632446,  2.35685668,  0.76146984,  8.41668302,  3.18580931,  4.04224808,
                            5.31969773,  9.16275397,  5.17652933,  6.11968075,  8.70230757,  6.42902741,
                            2.67848656,  2.71030438,  8.39292199,  8.24898974,  0.48159622,  5.79235008,
                            2.82793412,  5.21648905,  3.62208473,  1.71775652,  3.3032883,   7.32164841,
                            1.49429437,  3.74921596,  5.67211172,  0.48162585,  8.88405316,  0.53897842,
                            5.39172567,  2.48267169,  7.946164,    9.02791093,  0.56115775,  2.45772551,
                            8.26570773,  6.0443104,   1.6260163,   1.03051747,  2.82285864,  7.45984642,
                            3.36747846,  8.52111997]),[8,7])  # learning result
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



print('\n \nTotal elapsed time: {} sec'.format(time.time() - s))

