__author__ = 'enaghib'
import ephem
import numpy as np
import sqlite3 as lite
import Graphics as GP

Date = ephem.Date('2015/06/28 12:00:00.00')

con = lite.connect('FakeFBDE.db')
cur = con.cursor()
cur.execute('CREATE TABLE IF NOT EXISTS NightSummary('
                        'Night_count INTEGER, '
                        'T_start REAL, '
                        'T_end REAL, '
                        'Initial_field, '
                        'N_visits INTEGER, '
                        'N_triple INTEGER, '
                        'N_double INTEGER, '
                        'N_single INTEGER, '
                        'N_per_hour REAL, '
                        'Avg_cost REAL, '
                        'Avg_slew_t REAL, '
                        'Avg_alt REAL, '
                        'Avg_ha REAL)')

cur.execute('CREATE TABLE IF NOT EXISTS Schedule('
                        'Visit_count INTEGER, '
                        'Field_id INTEGER, '
                        'ephemDate REAL, '
                        'Filter INTEGER, '
                        'n_ton INTEGER, '
                        'n_previous INEGER, '
                        'Cost REAL, '
                        'Slew_t REAL, '
                        't_since_v_ton REAL,'
                        't_since_v_prev REAL,'
                        'Alt REAL, '
                        'HA REAL, '
                        't_to_invis REAL, '
                        'Sky_bri REAL, '
                        'Temp_coverage REAL)')

Time_slots = np.loadtxt("NightDataInLIS/TimeSlots{}.lis".format(int(ephem.julian_date(Date))), unpack = True)

Visit_count = 0
for i,t in enumerate(Time_slots):
    Visit_count  += 1
    Field_id      = 1
    ephemDate     = t
    Filter        = 0
    n_ton         = 0
    n_last        = 0
    Cost          = 0
    Slew_t        = 0
    t_since_v_ton = 0
    t_since_v_last= 0
    Alt           = 0
    HA            = 0
    t_to_invis    = 0
    Sky_bri       = 0
    Temp_coverage = 0

    cur.execute('INSERT INTO Schedule VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                                 (Visit_count, Field_id, ephemDate, Filter, n_ton, n_last, Cost, Slew_t/ephem.second, t_since_v_ton, t_since_v_last, Alt, HA, t_to_invis, Sky_bri, Temp_coverage))
con.commit()




Night_count   = 0
T_start       = Time_slots[0]
T_end         = Time_slots[-1]
Initial_field = 1
N_visits      = len(Time_slots)
N_triple    = 0
N_double    = 0
N_single    = 0

N_per_hour  = 0
Avg_cost    = 0
Avg_slew_t  = 0
Avg_alt     = 0
Avg_ha      = 0

cur.execute('INSERT INTO NightSummary VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                         (Night_count, T_start, T_end,  Initial_field, N_visits, N_triple, N_double, N_single, N_per_hour, Avg_cost, Avg_slew_t/ephem.second, Avg_alt, Avg_ha))
con.commit()

# Animation specifications
FPS = 10            # Frame per second
Steps = 100          # Simulation steps
MP4_quality = 300   # MP4 size and quality

PlotID = 1        # 1 for one Plot, 2 for including covering pattern
GP.Visualize(Date, PlotID ,FPS, Steps, MP4_quality, 'Visualizations/LSST1plot{}.mp4'.format(int(ephem.julian_date(Date))), showClouds= True, db = 'FakeFBDE.db')