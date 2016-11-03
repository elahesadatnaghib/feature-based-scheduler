__author__ = 'Elahe'


import sqlite3 as lite
import numpy as np
import ephem


''' Connect to the FBDE data base '''
def DBreadNwrite(key, Date):
    if key == 'w':
        FBDEcon = lite.connect('FBDE.db')
        FBDEcur = FBDEcon.cursor()

        # avoid overwrite
        try:
            FBDEcur.execute('SELECT * FROM NightSummary ORDER BY Night_count DESC LIMIT 1')
            last_row_ns = FBDEcur.fetchone()
            t_start_db = last_row_ns[1]
            t_end_db   = last_row_ns[2]
            if int(t_start_db + t_end_db)/2 == int(Date):
                print('This night is already recorded in the database')
                return
        except:
            print('Database created just now')

        # avoid dropping a night
        try:
            if int(t_start_db + t_end_db)/2 < int(Date) -1:
                print('One or more night(s) are missing')
                return
            if int(t_start_db + t_end_db)/2 > int(Date) -1:
                print('Last recorded night is after the intended night')
                return
        except:
            pass


        FBDEcur.execute('CREATE TABLE IF NOT EXISTS Schedule('
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

        FBDEcur.execute('CREATE TABLE IF NOT EXISTS NightSummary('
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

        FBDEcur.execute('CREATE TABLE IF NOT EXISTS FieldsStatistics('
                        'ID INTEGER, '
                        'Dec REAL, '
                        'RA REAL, '
                        'Fourth_last_visit REAL, '
                        'Third_last_visit REAL, '
                        'Second_last_visit REAL, '
                        'Last_visit REAL, '
                        'N_visit INTEGER, '
                        'Coadded_depth REAL, '
                        'Avg_cost REAL, '
                        'Avg_slew_t REAL, '
                        'Avg_alt REAL, '
                        'Avg_ha REAL)')

        FBDEcur.execute('CREATE TABLE IF NOT EXISTS Watch('
                        'Visit_count INTEGER, '
                        'ID INTEGER,'
                        'ephemDate,'
                        'F1,'
                        'F2,'
                        'F3,'
                        'F4,'
                        'F5,'
                        'F6,'
                        'F7)')

        Watch = np.load("Output/Watch{}.npy".format(int(ephem.julian_date(Date))))
        Schedule = np.load("Output/Schedule{}.npy".format(int(ephem.julian_date(Date))))
        Summary = np.load("Output/Summary{}.npy".format(int(ephem.julian_date(Date))))
        # 3 by n_fields matrix of ID, RA, Dec
        all_fields = np.loadtxt("NightDataInLIS/Constants/fieldID.lis", dtype = "i4, f8, f8", unpack = True)

        N_visits = np.count_nonzero(Schedule['Field_id'])




        ''' Update the SCHEDULE db'''
        # Import last row of the data base
        try:
            FBDEcur.execute('SELECT * FROM Schedule ORDER BY Visit_count DESC LIMIT 1')
            last_row_sch = FBDEcur.fetchone()
            Visit_count = last_row_sch[0]
        except:
            Visit_count = 0

        for index in range(N_visits):
            Visit_count  += 1
            Field_id      = Schedule[index]['Field_id']
            ephemDate     = Schedule[index]['ephemDate']
            Filter        = Schedule[index]['Filter']
            n_ton         = Schedule[index]['n_ton']
            n_last        = Schedule[index]['n_last']
            Cost          = Schedule[index]['Cost']
            Slew_t        = Schedule[index]['Slew_t']
            t_since_v_ton = Schedule[index]['t_since_v_ton']
            t_since_v_last= Schedule[index]['t_since_v_last']
            Alt           = Schedule[index]['Alt']
            HA            = Schedule[index]['HA']
            t_to_invis    = Schedule[index]['t_to_invis']
            Sky_bri       = Schedule[index]['Sky_bri']
            Temp_coverage = Schedule[index]['Temp_coverage']

            FBDEcur.execute('INSERT INTO Schedule VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                                 (Visit_count, Field_id, ephemDate, Filter, n_ton, n_last, Cost, Slew_t/ephem.second, t_since_v_ton, t_since_v_last, Alt, HA, t_to_invis, Sky_bri, Temp_coverage))
        FBDEcon.commit()

        ''' Update the NIGHT SUMMARY db'''
        # Import last row of the data base
        try:
            FBDEcur.execute('SELECT * FROM NightSummary ORDER BY Night_count DESC LIMIT 1')
            last_row_ns = FBDEcur.fetchone()
            Night_count = last_row_ns[0]
        except:
            Night_count = 0

        Night_count  += 1
        T_start       = Summary[0]
        T_end         = Summary[1]
        Initial_field = Summary[2]
        N_visits      = N_visits

        u, c           = np.unique(Schedule['Field_id'], return_counts=True)
        unique, counts = np.unique(c, return_counts=True)
        try:
            N_triple    = counts[unique == 3][0]
        except:
            N_triple    = 0
        try:
            N_double    = counts[unique == 2][0]
        except:
            N_double    = 0
        try:
            N_single    = counts[unique == 1][0]
        except:
            N_single    = 0

        N_per_hour  = N_visits * ephem.hour/ (T_end - T_start)
        Avg_cost    = np.average(Schedule[0:N_visits]['Cost'])
        Avg_slew_t  = np.average(Schedule[0:N_visits]['Slew_t'])
        Avg_alt     = np.average(Schedule[0:N_visits]['Alt'])
        Avg_ha      = np.average(Schedule[0:N_visits]['HA'])

        FBDEcur.execute('INSERT INTO NightSummary VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                                 (Night_count, T_start, T_end,  Initial_field, N_visits, N_triple, N_double, N_single, N_per_hour, Avg_cost, Avg_slew_t/ephem.second, Avg_alt, Avg_ha))
        FBDEcon.commit()



        ''' Update the FIELDS STATISTICS db'''
        try:
            FBDEcur.execute('SELECT * FROM FieldsStatistics ORDER BY ID DESC LIMIT 1')
            last_ID = FBDEcur.fetchone()[0]
        except: # Initialize FieldsStatistics
            for field in np.transpose(all_fields):
                ID = field[0]
                RA = field[1]
                Dec= field[2]
                FBDEcur.execute('INSERT INTO FieldsStatistics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                                 (ID, RA, Dec, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
                FBDEcon.commit()

        for index, id in enumerate(Schedule[0:N_visits]['Field_id']):
            FBDEcur.execute('SELECT * FROM FieldsStatistics WHERE ID = ?',(id,))
            field_row = FBDEcur.fetchone()
            previous_Fourth_last_visit = field_row[3] # thrown away
            previous_Third_last_visit  = field_row[4]
            previous_Second_last_visit = field_row[5]
            previous_Last_visit        = field_row[6]
            previous_N_visit           = field_row[7]
            previous_Coadded_depth     = field_row[8]
            previous_Avg_cost          = field_row[9]
            previous_Avg_slew_t        = field_row[10]
            previous_Avg_alt           = field_row[11]
            previous_Avg_ha            = field_row[12]

            Fourth_last_visit = previous_Third_last_visit
            Third_last_visit  = previous_Second_last_visit
            Second_last_visit = previous_Last_visit
            Last_visit        = Schedule[index]['ephemDate']
            N_visit           = previous_N_visit + 1
            Coadded_depth     = previous_Coadded_depth + 0 # temporarily
            Avg_cost          = 0#((previous_Avg_cost * previous_N_visit) + Schedule[index]['Cost'])/N_visit
            Avg_slew_t        = ((previous_Avg_slew_t * previous_N_visit) + Schedule[index]['Slew_t'])/N_visit
            Avg_alt           = ((previous_Avg_alt * previous_N_visit) + Schedule[index]['Alt'])/N_visit
            Avg_ha            = ((previous_Avg_ha * previous_N_visit) + Schedule[index]['HA'])/N_visit





            FBDEcur.execute('UPDATE FieldsStatistics SET '
                            'Fourth_last_visit = ?, '
                            'Third_last_visit  = ?, '
                            'Second_last_visit = ?, '
                            'Last_visit        = ?, '
                            'N_visit           = ?, '
                            'Coadded_depth     = ?, '
                            'Avg_cost          = ?, '
                            'Avg_slew_t        = ?, '
                            'Avg_alt           = ?, '
                            'Avg_ha            = ? WHERE ID = ?',
                            (Fourth_last_visit, Third_last_visit, Second_last_visit, Last_visit, N_visit, Coadded_depth, Avg_cost, Avg_slew_t/ephem.second, Avg_alt, Avg_ha, id))
        FBDEcon.commit()

        ''' Update the WATCH db'''
        # Import last row of the data base
        try:
            FBDEcur.execute('SELECT * FROM Watch ORDER BY Visit_count DESC LIMIT 1')
            last_row_sch = FBDEcur.fetchone()
            Visit_count = last_row_sch[0]
        except:
            Visit_count = 0

        for index in range(N_visits):
            Visit_count  += 1
            Field_id      = Watch[index]['Field_id']
            ephemDate     = Watch[index]['ephemDate']
            F1            = Watch[index]['F1']
            F2            = Watch[index]['F2']
            F3            = Watch[index]['F3']
            F4            = Watch[index]['F4']
            F5            = Watch[index]['F5']
            F6            = Watch[index]['F6']
            F7            = Watch[index]['F7']


            FBDEcur.execute('INSERT INTO Watch VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                                 (Visit_count, Field_id, ephemDate, F1, F2, F3, F4, F5, F6, F7))
        FBDEcon.commit()

        return

    if key == 'r':

        return


    return

'''
Date = ephem.Date('2016/09/01 12:00:00.00') # times are in UT
DBreadNwrite('w', Date)
'''