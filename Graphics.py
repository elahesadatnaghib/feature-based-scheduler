import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import ephem
import sqlite3 as lite
from progressbar import ProgressBar

# Altitude and Azimuth of a single field at t (JD) in rad
def Fields_local_coordinate(Field_ra, Field_dec, t, Site):

    # date and time
    Site.date = t
    curr_obj = ephem.FixedBody()
    curr_obj._ra = Field_ra * np.pi / 180
    curr_obj._dec = Field_dec * np.pi / 180
    curr_obj.compute(Site)
    altitude = curr_obj.alt
    azimuth = curr_obj.az
    return altitude, azimuth

def update_moon(t, Site) :
    Moon = ephem.Moon()
    Site.date = t
    Moon.compute(Site)
    X, Y = AltAz2XY(Moon.alt, Moon.az)
    r = Moon.size / 3600 * np.pi / 180  *2
    return X, Y, r, Moon.alt

def AltAz2XY(Alt, Az) :
    X = np.cos(Alt) * np.cos(Az) * -1
    Y = np.cos(Alt) * np.sin(Az)
    #Y = Alt * 2/ np.pi
    #X = Az / (2*np.pi)

    return Y, -1*X


def Visualize(Date, PlotID = 1,FPS = 15,Steps = 20,MP4_quality = 300, Name = "LSST Scheduler Simulator.mp4", showClouds = True):

    # Import data
    All_Fields = np.loadtxt("NightDataInLIS/Constants/fieldID.lis", unpack = True)
    N_Fields   = len(All_Fields[0])

    if showClouds:
        Time_slots = np.loadtxt("NightDataInLIS/TimeSlots{}.lis".format(int(ephem.julian_date(Date))), unpack = True)
        All_Cloud_cover = np.loadtxt("NightDataInLIS/Clouds{}.lis".format(int(ephem.julian_date(Date))), unpack = True)

    Site            = ephem.Observer()
    Site.lon        = -1.2320792
    Site.lat        = -0.517781017
    Site.elevation  = 2650
    Site.pressure   = 0.
    Site.horizon    = 0.

    #Initialize date and time
    lastN_start = float(Date) -1;   lastN_end = float(Date)
    toN_start = float(Date);        toN_end = float(Date) + 1

    #Connect to the History data base
    con = lite.connect('FBDE.db')
    cur = con.cursor()

    # Prepare to save in MP4 format
    FFMpegWriter = animation.writers['ffmpeg']
    metadata = dict(title='LSST Simulation', artist='Elahe', comment='Test')
    writer = FFMpegWriter(fps=FPS, metadata=metadata)

    #Progress bar initialization

    pbar = ProgressBar()

    # Initialize plot
    Fig = plt.figure()
    if PlotID == 1:
        ax = plt.subplot(111, axisbg = 'black')
    if PlotID == 2:
        ax = plt.subplot(211, axisbg = 'black')

    unobserved, Observed_lastN, Obseved_toN,\
    ToN_History_line, last_10_History_line,\
    Horizon, airmass_horizon, S_Pole,\
    LSST,\
    Clouds\
        = ax.plot([], [], '*',[], [], '*',[], [], '*',
                  [], [], '-',[], [], '*',
                  [], [], '-',[], [], '-',[], [], 'D',
                  [], [], 'o',
                  [], [], 'o')

    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal', adjustable = 'box')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Coloring
    Horizon.set_color('white'); airmass_horizon.set_color('red')
    S_Pole.set_markersize(3);   S_Pole.set_markerfacecolor('red')
    star_size = 4

    unobserved.set_color('dimgray');        unobserved.set_markersize(star_size)
    Observed_lastN.set_color('blue');       Observed_lastN.set_markersize(star_size)
    Obseved_toN.set_color('chartreuse');    Obseved_toN.set_markersize(star_size +2)
    Clouds.set_color('white');              Clouds.set_markersize(8)
    ToN_History_line.set_color('orange');   ToN_History_line.set_lw(.5)
    last_10_History_line.set_color('red');  last_10_History_line.set_lw(.5)

    LSST.set_color('red'); LSST.set_markersize(8)

    if PlotID == 2:
        freqAX = plt.subplot(212)
        cur.execute('SELECT N_visit, Last_visit, Second_last_visit, Third_last_visit, Fourth_last_visit From FieldsStatistics')
        row = cur.fetchall()
        N_visit     = [x[0] for x in row]
        Last_visit   = [x[1] for x in row]
        Second_last_visit = [x[2] for x in row]
        Third_last_visit  = [x[3] for x in row]
        Fourth_last_visit = [x[4] for x in row]

        initHistoricalcoverage = N_visit
        for index, id in enumerate(All_Fields):
            if Last_visit[index] > toN_start:
                initHistoricalcoverage[index] -= 1
                if Second_last_visit[index] > toN_start:
                    initHistoricalcoverage[index] -= 1
                    if Third_last_visit > toN_start:
                        initHistoricalcoverage[index] -= 1



        covering,current_cover = freqAX.plot(All_Fields[0],initHistoricalcoverage,'-',[],[],'o')

        freqAX.set_xlim(0,N_Fields)
        freqAX.set_ylim(0,np.max(initHistoricalcoverage)+5)
        covering.set_color('chartreuse');   covering.set_markersize(2)
        current_cover.set_color('red');     current_cover.set_markersize(6)



    cur.execute('SELECT Night_count, T_start, T_end FROM NightSummary WHERE T_start BETWEEN (?) AND (?)',(toN_start, toN_end))
    row = cur.fetchone()
    vID = row[0]
    t_start = row[1]
    t_end   = row[2]
    t = t_start


    # Figure labels and fixed elements
    Phi = np.arange(0, 2* np.pi, 0.05)
    Horizon.set_data(1.01*np.cos(Phi), 1.01*np.sin(Phi))
    ax.text(-1.3, 0, 'West', color = 'white', fontsize = 7)
    ax.text(1.15, 0 ,'East', color = 'white', fontsize = 7)
    ax.text( 0, 1.1, 'North', color = 'white', fontsize = 7)
    airmass_horizon.set_data(np.cos(np.pi/4) * np.cos(Phi), np.cos(np.pi/4) *  np.sin(Phi))
    ax.text(-.3, 0.6, 'Acceptable airmass horizon', color = 'white', fontsize = 5, fontweight = 'bold')
    Alt, Az = Fields_local_coordinate(180, -90, t, Site)
    x, y = AltAz2XY(Alt,Az)
    S_Pole.set_data(x, y)
    ax.text(x+ .05, y, 'S-Pole', color = 'white', fontsize = 7)

    # Observed last night fields
    cur.execute('SELECT Field_id FROM Schedule WHERE ephemDate BETWEEN (?) AND (?)',(lastN_start, lastN_end))
    row = cur.fetchall()
    if row is not None:
        F1 = [x[0] for x in row]
    else:
        F1 = []

    # Tonight observation path
    cur.execute('SELECT Field_id, ephemDate FROM Schedule WHERE ephemDate BETWEEN (?) AND (?)',(toN_start, toN_end))
    row = cur.fetchall()
    if row[0][0] is not None:
        F2 = [x[0] for x in row]
        F2_timing = [x[1] for x in row]
    else:
        F2 = []; F2_timing = []

    # Sky elements
    Moon = Circle((0, 0), 0, color = 'silver', zorder = 3)
    ax.add_patch(Moon)
    Moon_text = ax.text([], [], 'Moon', color = 'white', fontsize = 7)


    with writer.saving(Fig, Name, MP4_quality) :
        for t in pbar(np.linspace(t_start, t_end, num = Steps)):


            # Find the index of the current time
            time_index = 0
            while t > F2_timing[time_index]:
                time_index += 1
            if showClouds:
                Slot_n = 0
                while t > Time_slots[Slot_n]:
                    Slot_n += 1

            visit_index = 0
            visited_field = 0


            # Object fields: F1)Observed last night F2)Observed tonight F3)Unobserved F4)Covered by clouds
            F1_X = []; F1_Y = []; F2_X = []; F2_Y = []; F3_X = []; F3_Y = []; F4_X = []; F4_Y = []

            # F1  coordinate:
            for i in F1:
                Alt, Az = Fields_local_coordinate(All_Fields[1,i-1], All_Fields[2,i-1], t, Site)
                if Alt > 0:
                    X, Y    = AltAz2XY(Alt,Az)
                    F1_X.append(X); F1_Y.append(Y)

            # F2  coordinate:
            for i,tau in zip(F2,F2_timing):
                Alt, Az = Fields_local_coordinate(All_Fields[1,i-1], All_Fields[2,i-1], t, Site)
                if Alt > 0:
                    X, Y    = AltAz2XY(Alt,Az)
                    F2_X.append(X); F2_Y.append(Y)
                    if t >= tau:
                        visit_index = len(F2_X) -1
                        visited_field = i


            # F3  coordinate:
            for i in range(0,N_Fields):
                if True:
                    Alt, Az = Fields_local_coordinate(All_Fields[1,i], All_Fields[2,i], t, Site)
                    if Alt > 0:
                        X, Y    = AltAz2XY(Alt,Az)
                        F3_X.append(X); F3_Y.append(Y)

            # F4 coordinates
            if showClouds:
                for i in range(0,N_Fields):
                    if All_Cloud_cover[Slot_n, i] == 2:
                        Alt, Az = Fields_local_coordinate(All_Fields[1,i], All_Fields[2,i], t, Site)
                    if Alt > 0:
                        X, Y    = AltAz2XY(Alt,Az)
                        F4_X.append(X); F4_Y.append(Y)




            # Update plot
            unobserved.set_data([F3_X,F3_Y])
            Observed_lastN.set_data([F1_X,F1_Y])
            Obseved_toN.set_data([F2_X[0:visit_index],F2_Y[0:visit_index]])

            ToN_History_line.set_data([F2_X[0:visit_index], F2_Y[0:visit_index]])
            last_10_History_line.set_data([F2_X[visit_index - 10: visit_index], F2_Y[visit_index - 10: visit_index]])
            LSST.set_data([F2_X[visit_index],F2_Y[visit_index]])
            Clouds.set_data([F4_X,F4_Y])


            # Update Moon
            X, Y, r, alt = update_moon(t, Site)
            Moon.center = X, Y
            Moon.radius = r
            if alt > 0:
                #Moon.set_visible(True)
                Moon_text.set_visible(True)
                Moon_text.set_x(X+.002); Moon_text.set_y(Y+.002)
            else :
                Moon.set_visible(False)
                Moon_text.set_visible(False)

            #Update coverage
            if PlotID == 2:
                Historicalcoverage = np.zeros(N_Fields)
                for i,tau in zip(F2, F2_timing):
                    if tau <= t:
                        Historicalcoverage[i -1] += 1
                    else:
                        break
                tot = Historicalcoverage + initHistoricalcoverage
                current_cover.set_data(visited_field -1,tot[visited_field -1])
                covering.set_data(All_Fields[0], tot)

            #Observation statistics
            leg = plt.legend([Observed_lastN, Obseved_toN],
                       ['Visited last night', time_index])
            for l in leg.get_texts():
                l.set_fontsize(6)
            date = ephem.date(t)
            Fig.suptitle('Top view of the LSST site on {}, GMT'.format(date))


            '''
            # progress
            perc= int(100*(t - t_start)/(t_end - t_start))
            if perc <= 100:
                print('{} %'.format(perc))
            else:
                print('100 %')
            '''
            #Save current frame
            writer.grab_frame()



'''
Date = ephem.Date('2016/09/3 12:00:00.00') # times are in UT
# Animation specifications
FPS = 10            # Frame per second
Steps = 100          # Simulation steps
MP4_quality = 300   # MP4 size and quality

PlotID = 2        # 1 for one Plot, 2 for including covering pattern
Visualize(Date, PlotID ,FPS, Steps, MP4_quality, 'Visualizations/LSST1plot{}.mp4'.format(int(ephem.julian_date(Date))), showClouds= False)
'''