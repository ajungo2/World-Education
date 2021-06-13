import pandas as pd 
import numpy as np
import dash_table
import dash_html_components as html
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import os
from Preprocessing import *
from warnings import filterwarnings
from Preprocessing import final_df
filterwarnings('ignore')


######### Summary table info for country #########################

# file info:
num_countries_after_filter_countries = len(final_df['Country Code'].unique())
num_indicators_after_filter_countries = len(final_df['Indicator Code'].unique())


##############same format of summary table################################
# initialize list
summary_table_df_filter_country = [["Countries", num_countries_after_filter_countries], [
    "Indicators", num_indicators_after_filter_countries], ["Years", "1990-2014"]]
# Create the pandas DataFrame
summary_table_df_filter_country = pd.DataFrame(
    summary_table_df_filter_country, columns=['Variables', 'Observations'])


#


Title = html.Div([
    html.H1('Pre-process: Indicators & Countries',
            className="h1 text-center font-weight-bold"),
    html.Br([]),
    html.P(
        """Our data has more than 3000 indicators, a deeper analysis of them
            would be very exhaustive and escape from the objective of the course.
            In this section we have as an main objective the selection of 15
            indicators and the countries on which we are going to focus our report.
            For being able to do so, we applied the following strategies:""",
        className="row text-dark",
    ),
    html.Br([]),
    html.P(
        """1) For the indicators: we will focus on the variables that provide 
            more information (less null values) and, then, we are going to select 
            the ones that evaluate GPD, Population and PISA test.""",
        className="row text-dark",
    ),
    html.P(
        """2) Countries: We will list the top and bottom 6 countries that got 
            the highest and lowest marks in the last year of data, and, in the 
            same way, we will include the variables which measures the growth 
            of the indicator throught the years and, afterwards, we are goint to 
            select the top and bottom 6 of this list.""",
        className="row text-dark",
    ),
])

######################## INTRO indicators #############################################
Introd_ind = html.Div([
    html.Div(
        [
            html.H1("Our indicators", className="h2"),
            html.Br([]),
        ],
        className="product",
    )
])

#################Part indicator: top 20 overall indicators#################################
Row_1 = html.Div([
    html.Div([
        # column 1 table
        html.H2('Top 20 with more data'),
        dash_table.DataTable(
            data=sorted_ind_top_20.to_dict('records'),
            columns=[{"name": i, "id": i}
                     for i in sorted_ind_top_20.columns],
            id='top_20_list',
            style_cell_conditional=[{'textAlign': 'center'}],
            style_table={'overflowX': 'auto'},
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
        )
    ], style={'width': '100%'}),
    html.Div([
        # column 2 graph
        dcc.Graph(id='graph_top_20',
                  figure={
                      "data": [
                          go.Bar(x=sorted_ind_top_20["count"],
                                 y=sorted_ind_top_20["Indicator Code"],
                                 orientation='h',
                                 marker={
                              "color": "#151c97",
                              "line": {
                                  "color": "rgb(0, 0, 255)",
                                  "width": 2,
                              },
                          },
                              name="Top 20 indicators",
                          )]
                  }
                  )
    ], className="row w-100")
])


#################top 50 overall indicators#################################
Row_2 = html.Div([
    html.Div(
        [
            html.P(
                """We decided to work with PISA indicator test, 
                this exam measures student performance in reading, mathematics, and science literacy.
                It is conducted every 3 years. """,
                className="row bg-dark text-warning p-2 rounded",
            ),
        ],
        className="py-3 h5 ",
    )
])

Row_3 = html.Div([
    html.Div([
        # column 1 table
        html.H2('Top 10 PISA indicator', className='font-weight-bold text-center'),
        dash_table.DataTable(
            data=sorted_ind_pisa_top_10.to_dict('records'),
            columns=[{"name": i, "id": i}
                     for i in sorted_ind_pisa_top_10.columns],
            id='top_10_Pisa_indicators ',
            style_cell_conditional=[{'textAlign': 'center'}],
            style_table={'overflowX': 'auto'},
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
        )
    ], style={'width': '80%'}),
    html.Div([
        # column 2 graph
        dcc.Graph(id='graph_top_pisa_10',
                  figure={
                      "data": [
                          go.Bar(x=sorted_ind_pisa_top_10["count"],
                                 y=sorted_ind_pisa_top_10["Indicator Name"],
                                 orientation='h',
                                 marker={
                              "color": "#1e6091",
                              "line": {
                                  "color": "rgb(0, 0, 255)",
                                  "width": 2,
                              },
                          },
                              name="Top 10 PISA indicator",
                          )]
                  }
                  )
    ], className="row w-100")
])

##############Conclution and summary table for indicators#########################

concl_ind = html.Div([
    html.Div(
        [
            html.H3("Data after filtering selected indicators"),
            html.Br([]),
            html.P(
                """
                We decided to work with 16 indicators that contain information
                regarding the population by country (total, male, female),
                the population of the official age for primary education
                (total, male, female), the mean performance of PISA
                (mathematics, science and reading scale) and the GDP per capita.
                Important to mention that this test is conducted every 3 years. """,
                style={"color": "#000000"}
            ),
        ], style={'width': '100%', 'display': 'inline-block'}),
    html.Div([
        # column 2 table
        dash_table.DataTable(
            data=summary_table_df_filter_ind.to_dict('records'),
            columns=[{"name": i, "id": i}
                     for i in summary_table_df_filter_ind.columns],
            id='Summaryind',
            style_cell_conditional=[{'textAlign': 'center'}],
            style_table={'overflowX': 'auto'},
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
        )
    ], style={'width': '100%', 'display': 'inline-block'})
])


#Population, total, male, female
# Population of the official age for primary education total, male, female
# PISA: Mean performance on the mathematics scale. total, male, female
# PISA: Mean performance on the science scale,  total, male, female
# PISA: Mean performance on the reading scale,  total, male, female
# GDP per capita, PPP

######################## Part Countries: ##############################################

######################## INTRO country #############################################
Introd_country = html.Div([
    html.Div(
        [
            html.H1("Our Countries"),
            html.Br([]),
            html.H2("Top & Bottom 6- LO.PISA.MAT"),
            html.P(
                """We are going to select the top 6 countries according to 
                LO.PISA.MAT indicator in the 2012 year (the test is taken every 3 years). """,
                className="text-dark"
            ),
        ],
        className="product pt-5",
    )
])


########## TOP AND BOTTOM 6 Countries #################################

def create_card(title, content):
    card = dbc.Card(
        dbc.CardBody(
            [
                html.H4(title, className="card-title"),
                html.Br(),
                html.H2(content, className="card-subtitle"),
                html.Br(),
            ]
        ),
        color="#848484", inverse=True
    )
    return(card)


card2 = create_card("Top 6 Countries: ",
                    "China, Singapore, Hong Kong, Korea, Rep, Macao SAR, Japan")
card1 = create_card("Bottom 6 Countries: ",
                    "Peru, Indonesia, Qatar, Colombia, Jordan, Tunisia")

# put the cards into the row 4:
cards = dbc.Row([dbc.Col(id='card2', children=[card2], md=6),
                 dbc.Col(id='card1', children=[card1], md=6)])

Row_4 = html.Div([cards])


#####################################

bar_top6 = dcc.Graph(
    id="top6",
    figure={
        "data": [
            {
                'x': top6_highestPISA_2012['Country Name'],
                'y': top6_highestPISA_2012['2012'],
                'name':'top_6',
                'type':'bar',
                'marker':dict(color='#159790'),
            }],
        "layout": {
            "title": dict(text="Top 6 Country regarding LO.PISA.MAT",
                          font=dict(
                              size=20,
                              color='black')),
            "xaxis": dict(tickfont=dict(
                          color='black')),
            "yaxis": dict(tickfont=dict(
                          color='black')),
            "width": "2000",
            # "grid": {"rows": 0, "columns": 0},
            "annotations": [
                {
                    "font": {
                        "size": 20
                    },
                    "showarrow": False,
                    "text": "",
                    "x": 0.2,
                    "y": 0.2
                }
            ],
            "showlegend": False
        }
    }
)


bar_bot6 = dcc.Graph(
    id="bot6",
    figure={
        "data": [
            {
                'x': bottom6_lowestPISA_2012['Country Name'],
                'y': bottom6_lowestPISA_2012['2012'],
                'name':'bot_6',
                'type':'bar',
                'marker':dict(color='#97151c'),
            }],
        "layout": {
            "title": dict(text="Bottom 6 Country regarding LO.PISA.MAT",
                          font=dict(
                              size=20,
                              color='black')),
            "xaxis": dict(tickfont=dict(
                          color='black')),
            "yaxis": dict(tickfont=dict(
                          color='black')),
            "width": "2000",
            # "grid": {"rows": 0, "columns": 0},
            "annotations": [
                {
                    "font": {
                        "size": 20
                    },
                    "showarrow": False,
                    "text": "",
                    "x": 0.2,
                    "y": 0.2
                }
            ],
            "showlegend": False
        }
    }
)


Row_5 = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Top 6 Countries', children=[
            bar_top6
        ]),
        dcc.Tab(label='Bottom 6 Countries', children=[
            bar_bot6
        ])
    ])
], className="pt-5")

############## Growth rate: TOP AND BOTTOM 6 Countries #################################
######################## INTRO country #############################################
Introd_c_growth = html.Div([
    html.Div(
        [
            html.H2("Top & Bottom 6- growth rate in % 2000-2012",)
        ],
        className="product font-weight-bold",
    )
])
######################### cards #############################

card4 = create_card("Top 6 Countries: ",
                    "Peru, Brazil, Poland, Chile, Luxembourg, Israel")
card3 = create_card("Bottom 6 Countries: ",
                    "New Zealand, United Kingdom, Sweden, Australia, France, Iceland")

# put the cards into the row 4:
cards_overallgrowth = dbc.Row([dbc.Col(id='card4', children=[
                              card4], md=6), dbc.Col(id='card3', children=[card3], md=6)])

Row_6 = html.Div([cards_overallgrowth])


#####################################

# ranking:
scatter_ranking_top = dcc.Graph(
    id="rankingtop6",
    figure=go.Figure(data=go.Scatter(
        x=top6_overall['Country Name'],
        y=top6_overall['PISA.MAT_growth'],
        mode='markers',
        marker=dict(size=[100, 80, 65, 50, 30, 20],
                    color=[0, 1, 2, 3, 4, 5])))
)


scatter_ranking_bottom = dcc.Graph(
    id="rankingbot6",
    figure=go.Figure(data=go.Scatter(
        x=bottom6_overall['Country Name'],
        y=bottom6_overall['PISA.MAT_growth'],
        mode='markers',
        marker=dict(size=[100, 80, 65, 50, 30, 20],
                    color=[6, 7, 8, 9, 1, 2])))
)

# bar:

bar_top6_overall = dcc.Graph(
    id="topoverall6",
    figure=go.Figure(data=[
        go.Bar(name='2012',
               x=top6_overall['Country Name'], y=top6_overall['2012']),
        go.Bar(name='2000',
               x=top6_overall['Country Name'], y=top6_overall['2000'])
    ])
)


bar_bot6_overall = dcc.Graph(
    id="bottomoverall6",
    figure=go.Figure(data=[
        go.Bar(name='2012',
               x=bottom6_overall['Country Name'], y=bottom6_overall['2012']),
        go.Bar(name='2000',
               x=bottom6_overall['Country Name'], y=bottom6_overall['2000'])
    ])
)


Row_7 = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Ranking top 6- growth in %', children=[
            scatter_ranking_top
        ]),
        dcc.Tab(label='Top 6: 2012> 2000', children=[
            bar_top6_overall
        ]),
        dcc.Tab(label='Ranking bottom 6- growth in % ', children=[
            scatter_ranking_bottom
        ]),
        dcc.Tab(label='Bottom 6: 2012 < 2000', children=[
            bar_bot6_overall
        ])
    ])
], className="pt-5")


##############Conclution and summary table for indicators#########################
concl_country = html.Div([
    html.Div(
        [
            html.H3("Data after filtering selected countries"),
            html.Br([]),
            html.P(
                """We decided to work with 23 countries that are include in the 
                  top and bottom 6 of the last year and, as well, have the highest
                 and bottom growth rate from 2000 to 2012 regarding the mean performance of PISA.""",
            ),
        ], className="col-6 text-dark"
    ),
    html.Div([
        # column 2 table
        dash_table.DataTable(
            data=summary_table_df_filter_country.to_dict(
                'records'),
            columns=[{"name": i, "id": i}
                     for i in summary_table_df_filter_country.columns],
            id='Summarycountries',
            style_cell_conditional=[{'textAlign': 'center'}],
            style_table={'overflowX': 'auto'},
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
        )
    ], className="col-6 w-100")
], className="row")



HTML = html.Div(style={'backgroundColor': '#fefefe'},
                      children=[Title, Introd_ind, Row_1, Row_2, Row_3, concl_ind,
                                Introd_country, Row_4, Row_5,
                                Introd_c_growth, Row_6, Row_7,
                                concl_country])

