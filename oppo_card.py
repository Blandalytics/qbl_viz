import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import scipy as sp
import os
import time
import datetime
import urllib
from datetime import date
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import math
import pickle
import nfl_data_py as nfl
import re
from PIL import Image
import textwrap
import xgboost
np.random.seed(12)

logo_loc = 'https://github.com/Blandalytics/qbl_viz/blob/main/QB-List-Logo-wide.png?raw=true'
logo = Image.open(urllib.request.urlopen(logo_loc))
st.image(logo, width=200)

# Define major season dates
this_year = 2023
now = datetime.datetime.now()
today = datetime.datetime(now.year, now.month, now.day)
season_start = datetime.datetime(this_year, 9, 7)
first_sunday = datetime.datetime(this_year, 9, 10)
season_end = datetime.datetime(this_year+1, 1, 10)

# Calculate most recent week
week = math.ceil(((season_end if today > season_end else today) - season_start).days / 7)
week = max(week,1)
window_start = max(1,week-3)
recent_week = math.ceil(((season_end if today > season_end else today) - season_start).days / 7)
rolling_week = range(window_start,week+1)

team_colors = {'ARI':'#97233F','ATL':'#A71930','BAL':'#241773','BUF':'#00338D',
               'CAR':'#0085CA','CHI':'#00143F','CIN':'#FB4F14','CLE':'#FB4F14',
               'DAL':'#B0B7BC','DEN':'#002244','DET':'#046EB4','GB':'#24423C',
               'HOU':'#C9243F','IND':'#003D79','JAX':'#136677','KC':'#CA2430',
               'LA':'#002147','LAC':'#2072BA','LV':'#C4C9CC','MIA':'#0091A0',
               'MIN':'#4F2E84','NE':'#0A2342','NO':'#A08A58','NYG':'#192E6C',
               'NYJ':'#203731','PHI':'#014A53','PIT':'#FFC20E','SEA':'#7AC142',
               'SF':'#C9243F','TB':'#D40909','TEN':'#4095D1','WAS':'#FFC20F'}

team_secondary = {'ARI':'#FFB612','ATL':'#000000','BAL':'#9E7C0C',
                  'BUF':'#C60C30','CAR':'#101820','CHI':'#C83803',
                  'CIN':'#000000','CLE':'#311D00','DAL':'#041E42',
                  'DEN':'#FB4F14','DET':'#B0B7BC','GB':'#FFB612',
                  'HOU':'#A71930','IND':'#A2AAAD','JAX':'#9F792C',
                  'KC':'#FFB81C','LA':'#FFA300','LAC':'#FFC20E','LV':'#000000',
                  'MIA':'#FC4C02','MIN':'#FFC62F','NE':'#C60C30','NO':'#101820',
                  'NYG':'#A71930','NYJ':'#000000','PHI':'#A5ACAF',
                  'PIT':'#101820','SEA':'#002244','SF':'#B3995D',
                  'TB':'#FF7900','TEN':'#0C2340','WAS':'#773141'}

qbl_white = '#FEFEFE'
qbl_main = '#218559'

sns.set_theme(
    style={
        'axes.facecolor': qbl_white,
        'figure.facecolor':qbl_white,
        'legend.facecolor':qbl_white,
     }
    )

# Offensive Positions
offense_pos = ['QB','RB','WR','TE']

# Load Models
rush_model = pickle.load(open('models/rush_xTD_model.pkl', 'rb'))
rush_yard_model = pickle.load(open('models/rush_yard_model_class.pkl', 'rb'))
dropback_model = pickle.load(open('models/dropback_model.pkl', 'rb')) 
rec_model = pickle.load(open('models/rec_xTD_model.pkl', 'rb'))
yac_model = pickle.load(open('models/yac_model_class.pkl', 'rb'))
wopr_model = pickle.load(open('models/wopr_model.pkl', 'rb'))
cp_model = pickle.load(open('models/cp_model.pkl', 'rb'))
    
# xTD Calculation
def rush_xTD(yardline_100,defenders_in_box,num_DL,num_LB,num_DB,model=rush_model):
  return 1 - 1/(1+math.exp(model['intercept']
                           + (model['coefficients']['yardline_100'] * yardline_100)
                           + (model['coefficients']['yardline_100_2'] * yardline_100**2)
                           + (model['coefficients']['defenders_in_box'] * defenders_in_box)
                           + (model['coefficients']['num_DL'] * num_DL)
                           + (model['coefficients']['num_LB'] * num_LB)
                           + (model['coefficients']['num_DB'] * num_DB)))

def xRush_yards(df,model=rush_yard_model):
    df = pd.DataFrame(model.predict_proba(df[model.feature_names_in_]))
    df['test_prob'] = 0
    for col in df.columns[:-1]:
        df['yards_'+str(int(col-5))] = df[col].mul(col-5)
        df['test_prob'] += df[col]
        df['cumulative_prob_'+str(int(col-5))] = df['test_prob']
    
    df['xRush_yards'] = df[[x for x in df.columns if 'yards' in str(x)]].sum(axis=1)
    df['median_xRush_yards'] = df[[x for x in df.columns if 'cumulative' in str(x)]].apply(lambda y:np.argmax(y>0.5)-6, axis=1)
    return df['xRush_yards'].to_list()

def xyac(df,model=yac_model):
    df = pd.DataFrame(model.predict_proba(df[model.feature_names_in_]))
    df['test_prob'] = 0
    for col in df.columns[:-1]:
        df['yards_'+str(int(col-3))] = df[col].mul(col-3)
        df['test_prob'] += df[col]
        df['cumulative_prob_'+str(int(col-3))] = df['test_prob']
    
    df['xYAC'] = df[[x for x in df.columns if 'yards' in str(x)]].sum(axis=1)
    df['median_xYAC'] = df[[x for x in df.columns if 'cumulative' in str(x)]].apply(lambda y:np.argmax(y>0.5)-4, axis=1)
    return df['xYAC'].to_list()

# read in data from NFLfastR (for this year)
@st.cache_data(ttl=12*3600)
def load_data(year=this_year):
    return nfl.import_pbp_data([year], downcast=True, cache=False, alt_path=None), nfl.import_rosters([year])

data, rosters = load_data()
positions = rosters[['player_id','position']].set_index('player_id').to_dict()['position']
positions['00-0033357'] = 'TE' # Taysom Hill as a TE lol
full_names = rosters[['player_id','player_name']].set_index('player_id').to_dict()['player_name']
teams = rosters[['player_id','team']].set_index('player_id').to_dict()['team']

# Fantasy Point Values
score_values = {'complete_pass':1,
                'yards_gained':0.1,
                'touchdown':6}

### Transform the raw data
market_data = (data
               .loc[(data['season_type']=='REG') &
                    data['play_type'].isin(['pass','run']) &
                    data['down'].notna() &
                    (data['rusher'].notna() | 
                    (data['receiver'].notna() & 
                     data['air_yards'].notna())),
                    ['passer','rusher','receiver','fantasy_player_name','qtr',
                     'posteam','week','play_type','complete_pass','yards_gained',
                     'yardline_100','air_yards','cp','xyac_mean_yardage',
                     'touchdown','td_team','two_point_conv_result','ydstogo',
                     'fumble_lost','interception','rusher_id','receiver_id',
                     'passer_id','fantasy_player_id','offense_personnel',
                     'defenders_in_box','defense_personnel','pass_location',
                     'roof','posteam_type','down','qb_hit','shotgun',
                     'score_differential', 'half_seconds_remaining',
                     'home_timeouts_remaining', 'away_timeouts_remaining',
                     'wp','vegas_wp','yards_after_catch',
                     'run_gap','run_location']]
               .astype({
                   'down':'int',
                   'score_differential':'int',
                   'qtr':'int',
               })
               .copy()
               )

market_data['ydstogo'] = market_data['ydstogo'].clip(1,100)
market_data['yardline_100'] = market_data['yardline_100'].clip(1,100)

market_data['offense_personnel'] = market_data['offense_personnel'].fillna(market_data['offense_personnel'].mode()[0])
market_data['defense_personnel'] = market_data['defense_personnel'].fillna(market_data['defense_personnel'].mode()[0])
market_data['defenders_in_box'] = market_data['defenders_in_box'].fillna(market_data['defenders_in_box'].median())

# Remove defensive TDs
market_data['touchdown'] = np.where(market_data['posteam']==market_data['td_team'],market_data['touchdown'],0)
#Apply Fantasy Point Values
market_data['fantasy_points'] = market_data[['complete_pass',
                                             'yards_gained',
                                             'touchdown']].values.dot(np.array(list(score_values.values())))
                                                   
# Generate a position column
market_data['rusher_position'] = market_data['rusher_id'].map(positions)
market_data.loc[market_data['play_type']=='pass','rusher_position'] = None

market_data['receiver_position'] = market_data['receiver_id'].map(positions)
market_data.loc[market_data['play_type']!='pass','receiver_position'] = None

market_data['passer_position'] = market_data['passer_id'].map(positions)
market_data.loc[market_data['play_type']!='pass','passer_position'] = None

market_data['position'] = np.where(market_data['rusher_position'].isnull(), market_data['receiver_position'], market_data['rusher_position'])

market_data = (market_data
               .loc[market_data['position'].isin(['RB', 'TE', 'WR', 'QB'])]
               )

market_data['num_QB'] = 1
market_data.loc[market_data['offense_personnel'].str.contains('QB'),'num_QB'] = market_data.loc[market_data['offense_personnel'].str.contains('QB'),'offense_personnel'].str[0].astype('int')
for pos in ['RB','TE','WR']:
  market_data['num_'+pos] = market_data['offense_personnel'].str.findall('[0-9] '+pos).str[0].str[0].astype('int')
market_data['num_OL'] = 11 - market_data[['num_QB','num_RB','num_TE','num_WR']].sum(axis=1)

for pos in ['DL','LB','DB']:
  market_data['num_'+pos] = market_data['defense_personnel'].str.findall('[0-9] '+pos).str[0].str[0].astype('int')

roof_check = 1 if 'open' in market_data['roof'].unique() else 0

market_data.loc[market_data['run_location']=='middle','run_gap'] = 'middle'
market_data[['era_1','era_2','era_3']] = 0,0,1
market_data['yrds_to_first'] = market_data['air_yards'].sub(market_data['ydstogo'])
market_data['yrds_to_endzone'] = market_data['yardline_100'].sub(market_data['air_yards'])
market_data = pd.get_dummies(market_data, columns=['roof','posteam_type','down','qtr','run_gap'])
market_data['mid_pass'] = 0
market_data.loc[market_data['pass_location']=='middle','mid_pass'] = 1

# Fill in empty columns early in season
for roof_type in ['closed','open','dome','outdoor']:
    if 'roof_'+roof_type not in market_data.columns.values:
        market_data['roof_'+roof_type] = 0
if 'qtr_5' not in market_data.columns.values:
    market_data['qtr_5'] = 0

market_data['roof_retractable'] = market_data['roof_closed'] if roof_check==0 else market_data[['roof_closed','roof_open']].sum(axis=1)

market_data['dropback_pred'] = dropback_model.predict_proba(market_data[dropback_model.feature_names_in_])[:,1]
market_data.loc[market_data['play_type']=='pass','cp'] = cp_model.predict_proba(market_data.loc[market_data['play_type']=='pass',cp_model.feature_names_in_])[:,1]
market_data.loc[market_data['play_type']=='pass','xYAC'] = xyac(market_data.loc[market_data['play_type']=='pass'])
market_data.loc[market_data['play_type']=='run','xRush_yards'] = xRush_yards(market_data.loc[market_data['play_type']=='run'])

market_data.loc[market_data['xYAC']==market_data['yardline_100'],'xYAC'] = 0
market_data['modeled_yards'] = market_data['air_yards'].add(market_data['xYAC'])
market_data['xRec_yards'] = market_data['cp'].mul(market_data['modeled_yards'])

# Generate xTD Values
market_data['xTD'] = 0
market_data.loc[market_data['play_type']=='pass','xTD'] = (
    rec_model
    .predict_proba(market_data.loc[market_data['play_type']=='pass',
                                   rec_model.feature_names_in_])
    [:,1])

market_data.loc[market_data['play_type']=='run','xTD'] = (
    market_data.loc[market_data['play_type']=='run']
    .apply(lambda x: rush_xTD(x['yardline_100'],
                              x['defenders_in_box'],
                              x['num_DL'],
                              x['num_LB'],
                              x['num_DB'])
                              ,axis=1)
    )

market_data.sort_values('fantasy_points',ascending=False)

ind_df_ru = (market_data
             .loc[(market_data.rusher.isna()==False)]
             .groupby(['posteam','week','rusher_id','position'],
                      as_index=False)
             [['play_type','yards_gained','xRush_yards','touchdown','xTD',
               'fantasy_points']]
             .agg({
                 'play_type':'count',
                 'yards_gained':'sum',
                 'xRush_yards':'sum',
                 'touchdown':'sum',
                 'xTD':'sum',
                 'fantasy_points':'sum'
                 })
             )

team_df_ru = (market_data
              .loc[(market_data.rusher.isna()==False)]
              .groupby(['posteam','week'],
                       as_index=False)
              [['play_type','yards_gained','xTD']]
              .agg({
                  'play_type':'count',
                  'yards_gained':'sum',
                  'xTD':'sum'
                  })
              )

rusher_market = (ind_df_ru
                   .merge(team_df_ru,how='left',on=['posteam','week'])
                   .drop(columns=['yards_gained_y'])
                   .rename(columns={
                       'posteam':'team',
                       'touchdown':'touchdowns',
                       'play_type_x': 'carries',
                       'play_type_y': 'team_carries',
                       'yards_gained_x': 'ru_yards',
                       'xTD_x': 'ru_xTD',
                       'xTD_y': 'team_ru_xTD'
                       })
                   .astype({
                       'carries':'int',
                       'team_carries':'int',
                       'ru_yards':'int',
                       'touchdowns':'int',
                       'ru_xTD':'float',
                       'team_ru_xTD':'float'
                       })
                   )

rusher_market['carry_share'] = rusher_market['carries'].div(rusher_market['team_carries'])
rusher_market['ru_xTD_share'] = rusher_market['ru_xTD'].div(rusher_market['team_ru_xTD'])

ind_df_re = (market_data
             .loc[(market_data.receiver.isna()==False)]
             .groupby(['posteam','week','receiver_id','position'],
                      as_index=False)
             [['play_type','complete_pass','cp','yards_gained','air_yards',
               'modeled_yards', 'xRec_yards','touchdown','xTD']]
             .agg({
                 'play_type':'count',
                 'complete_pass':'sum',
                 'cp':'mean',
                 'yards_gained':'sum',
                 'air_yards':'sum',
                 'modeled_yards':'sum',
                 'xRec_yards':'sum',
                 'touchdown':'sum',
                 'xTD':'sum'
                 })
             )

team_df_re = (market_data
                 .loc[(market_data.receiver.isna()==False)]
                 .groupby(['posteam','week'],
                          as_index=False)
                 [['play_type','yards_gained','air_yards','xTD']]
                 .agg({
                     'play_type':'count',
                     'yards_gained':'sum',
                     'air_yards':'sum',
                     'xTD':'sum'
                     })
              )

receiver_market = (ind_df_re
                   .merge(team_df_re,how='left',on=['posteam','week'])
                   .drop(columns=['yards_gained_y'])
                   .rename(columns={
                       'posteam':'team',
                       'touchdown':'touchdowns',
                       'play_type_x': 'targets',
                       'play_type_y': 'team_targets',
                       'yards_gained_x': 're_yards',
                       'air_yards_x': 'air_yards',
                       'air_yards_y': 'team_air_yards',
                       'xTD_x': 're_xTD',
                       'xTD_y': 'team_re_xTD'
                       })
                   .astype({
                       'targets':'int',
                       'team_targets':'int',
                       're_yards':'int',
                       'air_yards':'int',
                       'team_air_yards':'int',
                       'touchdowns':'int'
                       })
                   )

receiver_market['target_share'] = receiver_market['targets'].div(receiver_market['team_targets'])
receiver_market['air_yard_share'] = receiver_market['air_yards'].div(receiver_market['team_air_yards'])
receiver_market['re_xTD_share'] = receiver_market['re_xTD'].div(receiver_market['team_re_xTD'])

receiver_market['team_target_index'] = 1
avg_targets = receiver_market.groupby(['team','week'])[['team_targets']].mean().mean()[0]
receiver_market['team_target_index'] = receiver_market.apply(lambda x: x['team_targets']/avg_targets, axis=1)

receiver_market['adj_target_share'] = receiver_market['target_share'].mul(receiver_market['team_target_index'])
receiver_market['adj_air_yard_share'] = receiver_market['air_yard_share'].mul(receiver_market['team_target_index'])
receiver_market['adj_re_xTD_share'] = receiver_market['re_xTD_share'].mul(receiver_market['team_target_index'])

# Average depth-of-target
receiver_market['aDOT'] = (receiver_market['air_yards']
                            .div(receiver_market['targets']))

# Receiver Air Conversion Ratio (air yards -> yards gained)
receiver_market['RACR'] = (receiver_market['re_yards']
                            .div(receiver_market['air_yards']))
lg_racr = receiver_market['re_yards'].sum() / receiver_market['air_yards'].sum()

# Weighted OPportunity Rating (combine target share and air yard share)
receiver_market['WOPR'] = (receiver_market['air_yard_share'].mul(.7)
                            .add(receiver_market['target_share'].mul(1.5)))

# WOPR, with xTD share (from https://colab.research.google.com/drive/13Jz0RkuZUoJDZRc0BGPXn10qJYzWboWT?usp=sharing)
for ppr_value in wopr_model.keys():
    receiver_market['new_WOPR_'+str(ppr_value)] = (receiver_market['adj_target_share']
                                                   .mul(wopr_model[ppr_value]['adj_target_share'])
                                                   .add(receiver_market['adj_air_yard_share']
                                                        .mul(wopr_model[ppr_value]['adj_air_yard_share']))
                                                   .add(receiver_market['adj_re_xTD_share']
                                                        .mul(wopr_model[ppr_value]['adj_re_xTD_share']))
                                                   .div(3)
                                                  )

# Big Play Index (air_yard and xTD per target, in fantasy points weights)
receiver_market['BPI'] = (receiver_market['aDOT']
                          .mul(lg_racr)
                          .mul(score_values['yards_gained'])
                          .add(receiver_market['re_xTD']
                               .div(receiver_market['targets'])
                               .mul(score_values['touchdown'])))

weekly_market = (rusher_market
                 .merge(receiver_market,
                    how='outer',
                    left_on=['rusher_id','week','team','position'],
                    right_on=['receiver_id','week','team','position'])
                 .fillna(0)
                 .rename(columns={'yards_x':'ru_yards',
                                  'yards_y':'re_yards',
                                  'touchdowns_x':'ru_TD',
                                  'touchdowns_y':'re_TD',
                                  'posteam':'team',
                                  'complete_pass':'receptions'})
                 .astype({
                     'carries':'int',
                     'targets':'int',
                     'ru_TD':'int',
                     're_TD':'int',
                     'ru_yards':'int',
                     're_yards':'int',
                     'receptions':'int',
                     'xRush_yards':'float',
                     'xRec_yards':'float'
                     }))

weekly_market['player_id'] = weekly_market.apply(lambda x: x['receiver_id'] if x['rusher_id']==0 else x['rusher_id'],axis=1)
weekly_market['player'] = weekly_market['player_id'].map(full_names)

weekly_market = weekly_market.drop(columns=['rusher_id','receiver_id'])

weekly_market['xRec'] = weekly_market['cp'].mul(weekly_market['targets'])
weekly_market['opps'] = weekly_market['carries'].add(weekly_market['targets'])
weekly_market['wOpps'] = weekly_market['carries'].add(weekly_market['targets'].mul(2))
weekly_market['xTD'] = weekly_market['ru_xTD'].add(weekly_market['re_xTD'])
weekly_market['OPPO'] = (
    weekly_market['xRush_yards'].mul(score_values['yards_gained']) # Don't have a good way to predict ru_yards (except for maybe a straight avg??)
    .add(weekly_market['xRec'].mul(score_values['complete_pass'])) # PPR value
    .add(weekly_market['xRec_yards'].mul(score_values['yards_gained'])) # Combination of air_yards+expected yac, multiplied by cp
    .add(weekly_market['xTD'].mul(score_values['touchdown'])) # Combined ru & re xTD
)

weekly_market['OPPO_0ppr'] = (
    weekly_market['xRush_yards'].mul(score_values['yards_gained']) # Don't have a good way to predict ru_yards (except for maybe a straight avg??)
    .add(weekly_market['xRec_yards'].mul(score_values['yards_gained'])) # Combination of air_yards+expected yac, multiplied by cp
    .add(weekly_market['xTD'].mul(score_values['touchdown'])) # Combined ru & re xTD
)

for stat in ['carries','targets','opps','wOpps','ru_xTD','re_xTD','xTD','OPPO','OPPO_0ppr']:
    weekly_market['team_'+stat] = (weekly_market
                                   .apply(lambda x: weekly_market.loc[(weekly_market['team']==x['team']) & 
                                                                      (weekly_market['week']==x['week']),
                                                                      stat]
                                          .sum(),
                                          axis=1)
                                  )
    weekly_market[stat+'_share'] = weekly_market[stat].div(weekly_market['team_'+stat])  

weekly_market['total_yards'] = weekly_market['ru_yards'].add(weekly_market['re_yards'])
weekly_market['TD'] = weekly_market['ru_TD'].add(weekly_market['re_TD']).astype('int')

weekly_market['FP'] = (
    weekly_market['total_yards']
    .mul(score_values['yards_gained']) # Don't have a good way to predict ru_yards (except for maybe a straight avg??)
    .add(weekly_market['receptions']
         .mul(score_values['complete_pass'])) # PPR value
    .add(weekly_market['TD']
         .mul(score_values['touchdown'])) # Combined ru & re xTD
    .round(1)
)

weekly_market['FP_0ppr'] = (
    weekly_market['total_yards']
    .mul(score_values['yards_gained']) # Don't have a good way to predict ru_yards (except for maybe a straight avg??)
    .add(weekly_market['TD']
         .mul(score_values['touchdown'])) # Combined ru & re xTD
    .round(1)
)

weekly_market['PAO'] = weekly_market['FP'].sub(weekly_market['OPPO'])
weekly_market['PAO_0ppr'] = weekly_market['FP_0ppr'].sub(weekly_market['OPPO_0ppr'])

# Season OPPO Frame (sum)
season_market = (weekly_market
                 .assign(week=1)
                 .groupby(['player','position','player_id'],
                          as_index=False)
                 .sum()
                 .rename(columns={'week':'games'})
                 )

season_market['team'] = season_market['player_id'].map(teams)

season_market['carry_share'] = season_market['carries'].div(season_market['team_carries'])
season_market['target_share'] = season_market['targets'].div(season_market['team_targets'])
season_market['opp_share'] = season_market['opps'].div(season_market['team_opps'])
season_market['wOpp_share'] = season_market['wOpps'].div(season_market['team_wOpps'])
season_market['ru_xTD_share'] = season_market['ru_xTD'].div(season_market['team_ru_xTD'])
season_market['re_xTD_share'] = season_market['re_xTD'].div(season_market['team_re_xTD'])
season_market['xTD_share'] = season_market['xTD'].div(season_market['team_xTD'])
season_market['FP_pg'] = season_market['FP'].div(season_market['games'])
season_market['OPPO_pg'] = season_market['OPPO'].div(season_market['games'])
season_market['OPPO_0ppr_pg'] = season_market['OPPO_0ppr'].div(season_market['games'])
season_market['PAO_pg'] = season_market['FP_pg'].sub(season_market['OPPO_pg'])
season_market['yards'] = season_market['ru_yards'].add(season_market['re_yards'])
season_market['xYards'] = season_market['xRush_yards'].add(season_market['xRec_yards'])

stat_dict = {
    'FP':['FP','OPPO','OPPO'],
    'ru_yards':['Rush Yards','xRush_yards','xRush Yards'],
    'ru_TD':['Rush TD','ru_xTD','xRush TD'],
    'receptions':['Catches','xRec','xCatches'],
    're_yards':['Rec Yards','xRec_yards','xRec Yards'],
    're_TD':['Rec TD','re_xTD','xRec TD'],
    'yards':['Yards','xYards','xYards'],
    'TD':['TD','xTD','xTD'],
}
def dist_plot(player,ax,team_color,team_alt_color,stat='FP',df=season_market):
    stat_val = df.loc[df['player']==player,stat].div(df['games']).mean()
    xstat_val = df.loc[df['player']==player,stat_dict[stat][1]].div(df['games']).mean()

    pos = df.loc[df['player']==player,'position'].item()

    sns.kdeplot(x=df.loc[df['position']==pos,stat_dict[stat][1]].div(df.loc[df['position']==pos,'games']),
                color='#218559',
                fill=True,
                alpha=0.25,
                cut=0,
                ax=ax)

    ax.axvline(xstat_val, ymax=0.575, color=team_color, linewidth=4)
    ax.text(xstat_val,ax.get_ylim()[1]*0.825,stat_dict[stat][2], ha='center', va='center', fontsize=12, weight=800, color=team_alt_color, bbox=dict(facecolor='white', alpha=1, edgecolor=team_color, linewidth=2))

    ax.axvline(stat_val, ymax=0.1, color='black', linewidth=4)
    ax.text(stat_val,ax.get_ylim()[1]*0.35,stat_dict[stat][0], ha='center', va='center', fontsize=12, color='black', bbox=dict(facecolor='white', alpha=1, edgecolor='grey'))

    ax_x_lim = ax.get_xlim()[1].copy()
    ax.set(xlabel=None, 
           ylabel=None, 
           xlim=(-ax_x_lim*0.05 if min(stat_val,xstat_val)>=ax_x_lim*0.11 else min(stat_val-ax_x_lim*0.11,-ax_x_lim*0.11),
                 max(xstat_val*1.115,stat_val*1.115,ax_x_lim)), 
           ylim=(0,
                 ax.get_ylim()[1]))
    ax.set_yticklabels([])
    ax.tick_params(left=False)

def bright_val(hex):
    hex = hex.replace('#','')
    rgb = tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))
    denom = 255 * np.sqrt(0.299 + 0.587 + 0.114)
    return np.sqrt(0.299 * rgb[0]**2 + 0.587 * rgb[1]**2 + 0.114 * rgb[2]**2) / denom

def qblist_card(player, df=season_market, team_logos=pd.read_csv('https://raw.githubusercontent.com/nflverse/nflverse-pbp/master/teams_colors_logos.csv')):
    fig = plt.figure(figsize=(8,8))

    team = df.loc[df['player']==player,'team'].item()
    team_color = team_logos.loc[team_logos['team_abbr']==team,'team_color'].item() if (bright_val(team_logos.loc[team_logos['team_abbr']==team,'team_color'].item())>0.2) or (team_logos.loc[team_logos['team_abbr']==team,'team_color2'].item()=='#000000') else team_logos.loc[team_logos['team_abbr']==team,'team_color2'].item()
    team_alt_color = team_logos.loc[team_logos['team_abbr']==team,'team_color'].item()# if bright_val(team_logos.loc[team_logos['team_abbr']==team,'team_color'].item())>0.2 else team_logos.loc[team_logos['team_abbr']==team,'team_color2'].item()

    # Parameters to divide card
    grid_height = 7 # 8 for individual stats
    grid_width = 7
    # Divide card into tiles
    grid = plt.GridSpec(grid_height, grid_width, wspace=0.1*2, hspace=0.6, width_ratios=[1]+[2.9/3]*3+[0.1]+[1]*2,
                        height_ratios=[1.75,.25]+[1]*(grid_height-3)+[0.1])

    name_ax = plt.subplot(grid[0,1:5])
    name_ax.text(0,0,textwrap.fill(player, 14, break_on_hyphens=False), 
                 ha='center', va='center', 
                 color=team_color, 
                 fontsize=25, weight=1000)
    name_ax.set(xlabel=None, xlim=(-1,1), ylabel=None, ylim=(-1,1))
    name_ax.set_xticklabels([])
    name_ax.set_yticklabels([])
    name_ax.tick_params(left=False, bottom=False)

    team_watermark = Image.open(urllib.request.urlopen(team_logos.loc[team_logos['team_abbr']==team,'team_wordmark'].item()))

    team_ax = plt.subplot(grid[0,5:])
    team_ax.imshow(team_watermark)
    team_ax.axis('off')

    qbl_ax = plt.subplot(grid[0,0])
    qbl_ax.set_facecolor(team_color)
    qbl_ax.imshow(Image.open('QB-List-Logo.png'))
    qbl_ax.axis('off')

    desc_ax = plt.subplot(grid[1,:])
    desc_ax.text(0,0,'Per-Game PPR Stats ({}; Wks {}-{})'.format(df.loc[df['player']==player,'position'].item(),market_data['week'].min(),market_data['week'].max()), 
                 ha='center', va='center', fontsize=20)
    desc_ax.set(xlabel=None, xlim=(-1,1), ylabel=None, ylim=(-1,1))
    desc_ax.set_xticklabels([])
    desc_ax.set_yticklabels([])
    desc_ax.tick_params(left=False, bottom=False)

    oppo_ax = plt.subplot(grid[2,:])
    dist_plot(player, oppo_ax, team_color, team_alt_color)

    rec_ax = plt.subplot(grid[3,:])
    dist_plot(player, rec_ax, team_color, team_alt_color, stat='receptions')

    ru_yard_ax = plt.subplot(grid[4,:])
    dist_plot(player, ru_yard_ax, team_color, team_alt_color, stat='yards')

    ru_td_ax = plt.subplot(grid[5,:])
    dist_plot(player, ru_td_ax, team_color, team_alt_color, stat='TD')

    # Author
    author_ax = plt.subplot(grid[6,:4])
    author_ax.text(-0.9,-1.5,'@Blandalytics', 
                 ha='left', va='top', 
                 fontsize=10, weight=500)
    author_ax.set(xlabel=None, xlim=(-1,1), ylabel=None, ylim=(-1,1))
    author_ax.set_xticklabels([])
    author_ax.set_yticklabels([])
    author_ax.tick_params(left=False, bottom=False)
    
    # Citation
    citation_ax = plt.subplot(grid[6,4:])
    citation_ax.text(0.9,-1.5,'Data via nflfastR', 
                 ha='right', va='top', 
                 fontsize=10, weight=500)
    citation_ax.set(xlabel=None, xlim=(-1,1), ylabel=None, ylim=(-1,1))
    citation_ax.set_xticklabels([])
    citation_ax.set_yticklabels([])
    citation_ax.tick_params(left=False, bottom=False)
    
    # Box the Card
    fig.add_artist(Line2D([0.115, 0.925], [0.87, 0.87], color=qbl_main, linewidth=2))
    fig.add_artist(Line2D([0.115, 0.115], [0.125, 0.87], color=qbl_main, linewidth=2))
    fig.add_artist(Line2D([0.925, 0.925], [0.125, 0.87], color=qbl_main, linewidth=2))
    fig.add_artist(Line2D([0.115, 0.925], [0.125, 0.125], color=qbl_main, linewidth=2))

    # Underline the Header
    fig.add_artist(Line2D([0.115, 0.925], [0.73, 0.73], color=qbl_main, linewidth=2)) # 0.78 for ind stats
    
    sns.despine(left=True, bottom=True)
    st.pyplot(fig)

# Player
players = season_market.sort_values('OPPO',ascending=False)['player'].to_list()
player = st.selectbox('Choose a player:', players)

qblist_card(player)
