import ephem
import numpy as np
from datetime import datetime
import sqlite3 as lite
import os.path


def set_data_range(lsst, date, tint):
    '''Return numpy array of dates between astronomical twilight'''
    ss = set_time(ephem.Date(twilightEve(lsst, date)))
    sr = set_time(ephem.Date(twilightMorn(lsst, date)))
    return np.arange(ss, sr, tint)

def set_time(dtime):
    '''Ses time to rounded value'''
    y, m ,d, hh, mm, ss = dtime.tuple()
    mm = mm - (mm % 5)
    return ephem.Date(datetime(y, m , d, hh, mm, 5, 0))

def sunset(site, date):
    '''Sunset in UTC'''
    site.horizon = 0.
    sun = ephem.Sun()
    site.date = date
    return site.next_setting(sun)

def sunrise(site, date):
    '''Sunset in UTC'''
    site.horizon = 0.
    sun = ephem.Sun()
    site.date = date
    return site.next_rising(sun)

def twilightEve(site, date):
    '''Start of twilight in UTC'''
    site.horizon = "-18."
    sun = ephem.Sun()
    site.date = date
    return site.next_setting(sun)

def twilightMorn(site, date):
    '''End of twilight in UTC'''
    site.horizon = "-18."
    sun = ephem.Sun()
    site.date = date
    return site.next_rising(sun)

def secz(alt):
    '''Compute airmass'''
    if alt < ephem.degrees('03:00:00'):
        alt = ephem.degrees('03:00:00')
    sz = 1.0/np.sin(alt) - 1.0
    xp = 1.0 + sz*(0.9981833 - sz*(0.002875 + 0.0008083*sz))
    return xp

def effectiveTime(airmass, extinction_coefficient=0.11):
    '''Calculate the effective exposure time given an airmass'''
    t_exp = 30.
    extinction = 10**((extinction_coefficient*(airmass - 1.))/2.5)
    return t_exp/(extinction*extinction)

def Night_data(date, site, dt = 10 * ephem.minute, airmassLimit = 1.4, genifexist = False):
    '''Generate visibility data for LSST fields given a date (UT)'''
    # define LSST site (angles in radians)
    DataExist = False

    if not genifexist and os.path.isfile('NightDataInLIS/Altitudes{}.lis'.format(int(ephem.julian_date(date)))):
        print('History independent data already exist for this night')
        DataExist = True


    lsst = site

    # define date and time interval for airmass, and airmass limits
    # times are in UT
    time_interval = dt
    time_range = set_data_range(lsst, date, time_interval)

    # initialize source and read positions from file
    source = ephem.FixedBody()
    data = np.loadtxt('NightDataInLIS/Constants/fieldID.lis', unpack=True)
    n_all_fields = len(data[0])

    #define output files and write headers
    if not DataExist:
        opConstraints = open("NightDataInLIS/AirmassConstraints{}.lis".format(int(ephem.julian_date(date))),"w")
        opConstraints.write("#JD_UT(rise), JD_UT(set)\n")

        opAlt = open("NightDataInLIS/Altitudes{}.lis".format(int(ephem.julian_date(date))),"w")

        opHang = open("NightDataInLIS/HourAngs{}.lis".format(int(ephem.julian_date(date))),"w")

        opMsep = open("NightDataInLIS/MoonSeps{}.lis".format(int(ephem.julian_date(date))),"w")

        opAz  = open("NightDataInLIS/Azimuths{}.lis".format(int(ephem.julian_date(date))),"w")
        ExpoH = open("NightDataInLIS/TimeSlots{}.lis".format(int(ephem.julian_date(date))),"w")
        [ExpoH.write("%12.6f "%(t))for t in time_range]
        ExpoH.write("\n")

    opNvis = open("NightDataInLIS/tot_N_visit{}.lis".format(int(ephem.julian_date(date))),"w")

    opTLastVis = open("NightDataInLIS/t_last_visit{}.lis".format(int(ephem.julian_date(date))),"w")



    if not DataExist:
        #calculate rise and set times (below airmass 1.4) from evening to morning twilight
        #calculate effective exposure time every 5 mins from evening to morning twilight
        length = time_range.shape
        airmass = np.zeros(length)
        ephemDates = np.zeros(length)
        altitudes = np.zeros(length)
        azimuths  = np.zeros(length)
        moonseps  = np.zeros(length)
        hourangs  = np.zeros(length)

        moon = ephem.Moon()


        for field, ra, dec in zip(data[0],data[1],data[2]):
            eq = ephem.Equatorial(np.radians(ra), np.radians(dec))
            source._ra = eq.ra
            source._dec = eq.dec
            source._epoch = eq.epoch
            source.compute(lsst)

            for i,t in enumerate(time_range):
                lsst.date = ephem.Date(t)
                source.compute(lsst)
                altitudes[i] = source.alt
                airmass[i] = secz(altitudes[i])
                ephemDates[i] = float(t)
                azimuths[i] = source.az

                hourangs[i] = (float(lsst.sidereal_time()) - float(source.ra))*12.0/np.pi
                if hourangs[i] > 12:
                    hourangs[i] = hourangs[i] - 24
                if hourangs[i] < -12:
                    hourangs[i] = hourangs[i] + 24

                moon.compute(lsst)
                moonseps[i] = ephem.separation(moon,source)

            [opAlt.write("%12.6f "%(a))for a in altitudes]
            opAlt.write("\n")

            [opHang.write("%12.6f "%(h))for h in hourangs]
            opHang.write("\n")

            [opMsep.write("%12.6f "%(m))for m in moonseps]
            opMsep.write("\n")

            [opAz.write("%12.6f "%(az))for az in azimuths]
            opAz.write("\n")

            #apply airmass limits and extract rise and set times
            inRange = np.where(airmass < airmassLimit)
            if (inRange[0].size):
                rising = inRange[0][0]
                setting = inRange[0][-1]
                opConstraints.write("%12.6f %12.6f\n"%(ephemDates[rising], ephemDates[setting],))
            else:
                opConstraints.write("0. 0.\n")

    try:
        FBDEcon = lite.connect('FBDE.db')
        FBDEcur = FBDEcon.cursor()
        FBDEcur.execute('SELECT T_start, T_end FROM NightSummary ORDER BY Night_count DESC LIMIT 1')
        mean_date = int(np.average(FBDEcur.fetchone()))
        if mean_date + 1 < int(date):
            print('There is no record of last night in the database')
            return
        FBDEcur.execute('SELECT Last_visit, N_visit FROM FieldsStatistics')
        input = FBDEcur.fetchall()
    except:
        print('\n No database of previous observations found')
        input = np.zeros(n_all_fields, dtype = [('last_v', np.float), ('N_vis', np.int)])

    for entry in input:
        # time of the last visits before the current night
        if entry[0] == 0.:  # temporarily
            opTLastVis.write("%12.6f \n"%(1e10))
        else:
            opTLastVis.write("%12.6f \n"%(int(entry[0])))
        # total number of visits of each field before current night observation
        opNvis.write("%12.6f \n"%(entry[1]))


# Simulation specific: generating sky brightness and cloud coverage data
def gen_uncertain():
    con = lite.connect('Cloud.db')
    cur = con.cursor()
