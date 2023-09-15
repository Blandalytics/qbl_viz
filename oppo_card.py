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
import matplotlib.patheffects as patheffects
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
st.image(logo, width=250)

st.title("OPPO")
st.write('OPPO is a metric designed to combine multiple models (xRush Yards, xRush TDs, xCompletion%, xReceiving Yards, and xReceiving TDs) into an all-in-one fantasy metric. Standard PPR values are utilized (0.1 pts per yard, 6pts per TD, and 1pt per reception).')
st.write('For more information, read [this explainer article](https://football.pitcherlist.com/oppo-valuing-opportunities-for-fantasy-football/).')
st.write('Note: OPPO does ***not*** include passing stats.')
# Define major season dates
this_year = st.radio('Choose a season:', 
                     [2023,2022,2021,2020],
                     horizontal=True)
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

@st.cache_data(ttl=2*3600)
def load_data(year):
    return pd.read_csv(f'https://github.com/Blandalytics/qbl_viz/blob/main/data/season_market_data_{year}.csv?raw=true')
season_market = load_data(this_year)

stat_dict = {
    'FP':['Pts','OPPO','OPPO'],
    'ru_yards':['Rush Yards','xRush_yards','xRush Yards'],
    'ru_TD':['Rush TD','ru_xTD','xRush TD'],
    'receptions':['Catches','xRec','xCatches'],
    're_yards':['Rec Yards','xRec_yards','xRec Yards'],
    're_TD':['Rec TD','re_xTD','xRec TD'],
    'yards':['Yards','xYards','xYards'],
    'TD':['TD','xTD','xTD'],
}

def bright_val(hex):
    hex = hex.replace('#','')
    rgb = tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))
    denom = 255 * np.sqrt(0.299 + 0.587 + 0.114)
    return np.sqrt(0.299 * rgb[0]**2 + 0.587 * rgb[1]**2 + 0.114 * rgb[2]**2) / denom

def get_luminance(hex_color):
    color = hex_color[1:]

    hex_red = int(color[0:2], base=16)
    hex_green = int(color[2:4], base=16)
    hex_blue = int(color[4:6], base=16)

    return hex_red * 0.2126 + hex_green * 0.7152 + hex_blue * 0.0722

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
    ax.text(xstat_val,ax.get_ylim()[1]*0.825,stat_dict[stat][2], ha='center', va='center', fontsize=12, weight=800,
                        color=team_alt_color, bbox=dict(facecolor='w', alpha=1, edgecolor=team_color, linewidth=2))
  
    ax.axvline(stat_val, ymax=0.1, color='black', linewidth=4)
    ax.text(stat_val,ax.get_ylim()[1]*0.35,stat_dict[stat][0], ha='center', va='center', fontsize=12, color='black', bbox=dict(facecolor='white', alpha=1, edgecolor='grey'))

    ax_x_lim = ax.get_xlim()[1].copy()
    x_fudge = 0.08
    ax.set(xlabel=None, 
           ylabel=None, 
           xlim=(-ax_x_lim*0.05 if min(stat_val,xstat_val)>=ax_x_lim*x_fudge else min(stat_val-ax_x_lim*x_fudge,-ax_x_lim*x_fudge),
                 max(xstat_val*(1+x_fudge),stat_val*(1+x_fudge),ax_x_lim*1.05)), 
           ylim=(0,
                 ax.get_ylim()[1]))
    ax.set_yticklabels([])
    if stat=='FP':
      ax.set_xticks(np.arange(ax_x_lim+1)[::5])
    elif stat=='receptions':
      ax.set_xticks(np.arange(ax_x_lim+1))
    elif stat=='yards':
      ax.set_xticks(np.arange(ax_x_lim+1)[::20])
    else:
      ax.set_xticks([x/4 for x in np.arange(int(ax_x_lim*4+1))])
    ax.tick_params(left=False)

def qblist_card(player, df=season_market, team_logos=pd.read_csv('https://raw.githubusercontent.com/nflverse/nflverse-pbp/master/teams_colors_logos.csv')):
    fig = plt.figure(figsize=(8,8))

    team = df.loc[df['player']==player,'team'].item()
    team_color = team_logos.loc[team_logos['team_abbr']==team,'team_color'].item()# if (bright_val(team_logos.loc[team_logos['team_abbr']==team,'team_color'].item())>0.2) or (team_logos.loc[team_logos['team_abbr']==team,'team_color2'].item()=='#000000') else team_logos.loc[team_logos['team_abbr']==team,'team_color2'].item()
    team_alt_color = team_logos.loc[team_logos['team_abbr']==team,'team_color2'].item() if get_luminance(team_logos.loc[team_logos['team_abbr']==team,'team_color2'].item()) < 140 else team_logos.loc[team_logos['team_abbr']==team,'team_color'].item()
  
    # Parameters to divide card
    grid_height = 7 # 8 for individual stats
    grid_width = 7
    # Divide card into tiles
    grid = plt.GridSpec(grid_height, grid_width, wspace=0.1*2, hspace=0.6, width_ratios=[1]+[2.9/3]*3+[0.1]+[1]*2,
                        height_ratios=[1.75,.25]+[1]*(grid_height-3)+[0.1])

    name_ax = plt.subplot(grid[0,1:5])
    name_text = name_ax.text(0,0,textwrap.fill(player, 14, break_on_hyphens=False), 
                 ha='center', va='center', 
                 color=team_alt_color, 
                 fontsize=26, weight=1000)
    name_text.set_path_effects([patheffects.withStroke(linewidth=1.5, 
                                                       foreground=team_logos.loc[team_logos['team_abbr']==team,'team_color'].item() if team_color != team_alt_color else team_logos.loc[team_logos['team_abbr']==team,'team_color2'].item())])
    name_ax.set(xlabel=None, xlim=(-1,1), ylabel=None, ylim=(-1,1))
    name_ax.set_xticklabels([])
    name_ax.set_yticklabels([])
    name_ax.tick_params(left=False, bottom=False)

    team_watermark = Image.open(urllib.request.urlopen(team_logos.loc[team_logos['team_abbr']==team,'team_wordmark'].item()))

    team_ax = plt.subplot(grid[0,5:])
    team_ax.imshow(team_watermark)
    team_ax.axis('off')

    qbl_ax = plt.subplot(grid[0,0], facecolor=(team_color))
    qbl_ax.set_facecolor(team_color)
    qbl_ax.imshow(Image.open('QB-List-Logo.png'))
    qbl_ax.axis('off')

    desc_ax = plt.subplot(grid[1,:])
    week_text = f'Wks 1-{recent_week}' if recent_week!=1 else 'Wk 1'
    desc_ax.text(0,0,'Per-Game PPR Stats ({}; {})'.format(df.loc[df['player']==player,'position'].item(),week_text), 
                 ha='center', va='center', fontsize=20)
    desc_ax.set(xlabel=None, xlim=(-1,1), ylabel=None, ylim=(-1,1))
    desc_ax.set_xticklabels([])
    desc_ax.set_yticklabels([])
    desc_ax.tick_params(left=False, bottom=False)

    oppo_ax = plt.subplot(grid[2,:])
    dist_plot(player, oppo_ax, team_color, team_alt_color)

    rec_ax = plt.subplot(grid[3,:])
    dist_plot(player, rec_ax, team_color, team_alt_color, stat='receptions')

    yard_ax = plt.subplot(grid[4,:])
    dist_plot(player, yard_ax, team_color, team_alt_color, stat='yards')

    td_ax = plt.subplot(grid[5,:])
    dist_plot(player, td_ax, team_color, team_alt_color, stat='TD')

    # Author
    author_ax = plt.subplot(grid[6,:2])
    author_ax.text(-0.9,-1.5,'@Blandalytics', 
                 ha='left', va='top', 
                 fontsize=10, weight=500)
    author_ax.set(xlabel=None, xlim=(-1,1), ylabel=None, ylim=(-1,1))
    author_ax.set_xticklabels([])
    author_ax.set_yticklabels([])
    author_ax.tick_params(left=False, bottom=False)
  
    # Website
    website_ax = plt.subplot(grid[6,2:5])
    website_ax.text(0,-1.5,'qblist-oppo-card.streamlit.app', 
                 ha='center', va='top', 
                 fontsize=10, weight=500)
    website_ax.set(xlabel=None, xlim=(-1,1), ylabel=None, ylim=(-1,1))
    website_ax.set_xticklabels([])
    website_ax.set_yticklabels([])
    website_ax.tick_params(left=False, bottom=False)
    
    # Citation
    citation_ax = plt.subplot(grid[6,5:])
    citation_ax.text(0.9,-1.5,'Data via nflfastR', 
                 ha='right', va='top', 
                 fontsize=10, weight=500)
    citation_ax.set(xlabel=None, xlim=(-1,1), ylabel=None, ylim=(-1,1))
    citation_ax.set_xticklabels([])
    citation_ax.set_yticklabels([])
    citation_ax.tick_params(left=False, bottom=False)
    
    # Box the Card
    fig.add_artist(Line2D([0.115, 0.925], [0.87, 0.87], color=qbl_main, linewidth=2))
    fig.add_artist(Line2D([0.115, 0.115], [0.12, 0.87], color=qbl_main, linewidth=2))
    fig.add_artist(Line2D([0.925, 0.925], [0.12, 0.87], color=qbl_main, linewidth=2))
    fig.add_artist(Line2D([0.115, 0.925], [0.12, 0.12], color=qbl_main, linewidth=2))

    # Underline the Header
    fig.add_artist(Line2D([0.115, 0.925], [0.74, 0.74], color=qbl_main, linewidth=2)) # 0.78 for ind stats
    
    sns.despine(left=True, bottom=True)
    st.pyplot(fig)

pos_select = st.radio('Position Group', 
                      ['All','Flex']+offense_pos,
                      index=0,
                      horizontal=True
                      )

if pos_select=='All':
    pos_filter = offense_pos
elif pos_select=='Flex':
    pos_filter = ['RB','WR','TE']
else:
    pos_filter = [pos_select]

# Player
players = season_market.loc[season_market['position'].isin(pos_filter)].sort_values('OPPO',ascending=False)['player'].to_list()
player = st.selectbox('Choose a player:', players)

qblist_card(player)
