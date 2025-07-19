import pandas as pd
import pytz
def get_max_P(V, Srated=1, v1=253, v2=260):
    if V < v1:
        return Srated
    elif V > v2:
        return .2 * Srated
    else:
        m = (Srated - .2*Srated) / (v1 - v2)
        P = m * (V - v2) + .2*Srated
        return P
    
def get_voltvar_Q(V, Srated=1, v1=207, v2=220, v3=240, v4=258, Q1=.44, Q4=.60):
    if V <= v1:
        Q = Q1* Srated
    elif v1 <= V < v2:
        m = (Q1* Srated - 0) / (v1 - v2)
        Q = m * (V - v2)
    elif v2 <= V <= v3:
        Q = 0
    elif v3 < V < v4:
        m = (0 - Q4* Srated) / (v3 - v4)
        Q = -m * (V - v4) - Q4* Srated
    else:  # V >= v4
        Q = - Q4* Srated
    return Q

def process_edp(df, meta_data, offset=9.5*60):
    df5 = df.query("circuit_label == 'pv_site_net'")
    df5.loc[df5['voltage_avg'] > 300, 'voltage_avg'] = None
    # df5.loc[df5['voltage_avg'] > 300, 'voltage_avg'] = df5['voltage_avg']/1000
    df5 = df5.merge(meta_data, on='edp_site_id', how='left')
    df5 = df5[['edp_site_id', 'datetime', 'real_energy', 
        'reactive_energy', 'current_avg', 'voltage_avg', 
            'postcode', 'Srated']]
    df5['datetime'] = pd.to_datetime(df5['datetime'])
    df5['time'] = df5.pop('datetime')
    df5['time'] = df5['time'].dt.tz_localize(pytz.FixedOffset(offset))
    df5 = df5.groupby(['time', 'edp_site_id', 'postcode', 'Srated']).agg({'real_energy': 'sum', 'reactive_energy': 'sum', 
                    'current_avg': 'sum',  'voltage_avg': 'mean'}).reset_index()
    df5['active_power'] = df5['real_energy']*12/1000
    df5['reactive_power'] = df5['reactive_energy']*12/1000
    df5['apparent_power'] = df5['active_power']**2 + df5['reactive_power']**2
    df5['apparent_power'] = df5['apparent_power']**.5 
    df5['P_threshold'] = df5.apply(lambda row: get_max_P(row['voltage_avg'], Srated=row['Srated'], v1=253, v2=260), axis=1)
    df5['P_noncomp'] =  df5['active_power'] - df5['P_threshold']
    df5['P_noncomp'] = df5['P_noncomp'].clip(lower=0)
    df5['Q_voltvar'] =df5.apply(lambda row: get_voltvar_Q(row['voltage_avg'], Srated=row['Srated']), axis=1)
    df5['Q_voltvar_max'] = df5['Q_voltvar'] + df5['Srated'] * 0.05
    df5['Q_voltvar_min'] = df5['Q_voltvar'] - df5['Srated'] * 0.05
    df5['Q_noncomp'] = ((df5['reactive_power'] - df5['Q_voltvar']).abs() - df5['Srated'] * 0.05).abs()
    wrong_meta_data_based_on_maxP = df5.query(f"active_power > Srated")['edp_site_id'].unique()
    df5['wrong_on_maxP'] = False
    df5.loc[df5['edp_site_id'].isin(wrong_meta_data_based_on_maxP) == True, 'wrong_on_maxP'] = True
    df5 = df5.sort_values(by=['edp_site_id', 'time']).reset_index(drop=True)
    return df5