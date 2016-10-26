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


def Visualize(t_start, t_end, t_start_past, t_end_past, PlotID = 1,FPS = 15,Steps = 20,MP4_quality = 300, Name = "LSST Scheduler Simulator.mp4", showClouds = True):

    # Import data
    All_Fields = np.loadtxt("NightDataInLIS/Constants/fieldID.lis", unpack = True)
    N_Fields   = len(All_Fields[0])


    #Connect to the data base
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
    Clouds.set_color('white');              Clouds.set_markersize(10)

    ToN_History_line.set_color('orange');   ToN_History_line.set_lw(.5)
    last_10_History_line.set_color('red');  last_10_History_line.set_lw(.5)

    LSST.set_color('red'); LSST.set_markersize(8)

    if PlotID == 2:
        freqAX = plt.subplot(212)
        initHistoricalcoverage = np.zeros(N_Fields) #to be changed
        covering,current_cover = freqAX.plot(All_Fields[0],initHistoricalcoverage,'-',[],[],'o')

        freqAX.set_xlim(0,N_Fields)
        freqAX.set_ylim(0,np.max(initHistoricalcoverage)+5)
        covering.set_color('chartreuse');   covering.set_markersize(2)
        current_cover.set_color('red');     current_cover.set_markersize(6)



    Site            = ephem.Observer()
    Site.lon        = -1.2320792
    Site.lat        = -0.517781017
    Site.elevation  = 2650
    Site.pressure   = 0.
    Site.horizon    = 0.



    cur.execute('SELECT Visit_count, Field_id, ephemDate, Filter, n_ton, n_previous, Temp_coverage FROM Schedule WHERE ephemdate BETWEEN (?) AND (?)',(t_start, t_end))
    row = cur.fetchall()
    if row[0][0] is not None:
        Visit_count   = [x[0] for x in row]
        Field_id      = [x[1] for x in row]
        ephemDate     = [x[2] for x in row]
        Filter        = [x[3] for x in row]
        n_ton         = [x[4] for x in row]
        n_previous    = [x[5] for x in row]
        Temp_coverage = [x[6] for x in row]
    else:
        Visit_count   = []
        Field_id      = []
        ephemDate     = []
        Filter        = []
        n_ton         = []
        n_previous    = []
        Temp_coverage = []

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
    cur.execute('SELECT Field_id FROM Schedule WHERE ephemDate BETWEEN (?) AND (?)',(t_start_past, t_end_past))
    row = cur.fetchall()
    if row is not None:
        F1 = [x[0] for x in row]
    else:
        F1 = []

    # Tonight observation path
    F2        = Field_id
    F2_timing = ephemDate

    # Sky elements
    Moon = Circle((0, 0), 0, color = 'silver', zorder = 3)
    ax.add_patch(Moon)
    Moon_text = ax.text([], [], 'Moon', color = 'white', fontsize = 7)


    with writer.saving(Fig, Name, MP4_quality) :
        for t in pbar(np.linspace(t_start, t_end, num = Steps)):


            # Find the index of the current time
            time_index = 0
            temp = F2_timing[0]
            while t > temp:
                time_index += 1
                try:
                    temp = F2_timing[time_index]
                except:
                    time_index -= 1
                    temp = F2_timing[time_index]
                    break

            visit_index   = 0
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
            '''
            # F4 coordinates (temporary coverage)
            if showClouds:
                for i in range(0,N_Fields):
                    if All_Cloud_cover[i, Slot_n] == 1:
                        Alt, Az = Fields_local_coordinate(All_Fields[1,i], All_Fields[2,i], t, Site)
                    if Alt > 0:
                        X, Y    = AltAz2XY(Alt,Az)
                        F4_X.append(X); F4_Y.append(Y)
            '''



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
                current_cover.set_data(visited_field -1,1)
                covering.set_data(All_Fields[0],Historicalcoverage)

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






t_start_past = ephem.Date('2016/09/2 12:00:00.00')
t_end_past   = ephem.Date('2016/09/10 12:00:00.00')
t_start      = ephem.Date('2016/09/10 12:00:00.00')
t_end        = ephem.Date('2016/09/15 12:00:00.00')
# Animation specifications
FPS = 10            # Frame per second
Steps = 500          # Simulation steps
MP4_quality = 100   # MP4 size and quality

PlotID = 2          # 1 for one Plot, 2 for including covering pattern
Visualize(t_start, t_end, t_start_past, t_end_past, PlotID ,FPS, Steps, MP4_quality, 'Visualizations/LSST1plot{}.mp4'.format(int(ephem.julian_date(t_start))), showClouds= False)
