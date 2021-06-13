from jupyter_dash import JupyterDash
import pandas as pd
import matplotlib.pyplot as unique
import numpy as np
import dash_table
import dash_html_components as html
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objs as go
# from PlotlyApp import app
from Preprocessing import final_df
import plotly.express as px
from warnings import filterwarnings
filterwarnings('ignore')



# selected_variables = ["Country Code","Country Name", "Indicator Code", "Indicator Name", years from 1990 to 2014]

final_df = final_df[final_df.columns[0:29]]

# pivot from wide to long:

final_df = final_df.melt(
    id_vars=["Country Name", "Country Code",
             "Indicator Name", "Indicator Code"],
    value_vars=['1990', '1991', '1992', '1993', '1994', '1995',
                '1996', '1997', '1998', '1999', '2000', '2001',
                '2001', '2001', '2002', '2003', '2004', '2005',
                '2006', '2007', '2008', '2009', '2010', '2011',
                '2012', '2013', '2014'], var_name='Years', value_name='Values')


# Passing values to integer:

final_df['Years'] = final_df['Years'].astype(int)

# Delete the NAN Values
final_df = final_df.dropna()


# Growth population: total population VS years for the "SP.POP.TOTL.MA.IN"
filter_popind_final_df = final_df[(
    final_df["Indicator Code"] == "SP.POP.TOTL.MA.IN")]


# Growth population: total population VS years for the "SP.POP.TOTL.MA.IN"

filter_pop3ind_final_df = final_df[(final_df["Indicator Code"] == "SP.POP.TOTL.FE.IN") |
                                   (final_df["Indicator Code"] == "SP.POP.TOTL.MA.IN")]



Title_eda = html.Div([
    html.H1('Exploratory Data Analysis',
            className="text-center font-weight-bold display-4"),
    html.P(
        "Now, our data  has 16 indicators and 23 countries,",
        className="text-dark",
    ),
])

######################## INTRO indicators #############################################
Introd_eda = html.Div([
    html.Div(
        [
            html.H2("Checking relation between indicators"),
            html.Br([]),
        ],
        className="product",
    )
])

################# Overall #################################

available_indicators = final_df['Indicator Name'].unique()

Row_1 = html.Div([
    html.Div([

        html.Div([
            dcc.Dropdown(
                id='crossfilter-xaxis-column',
                options=[{'label': i, 'value': i}
                         for i in available_indicators],
                value='Population of the official age for pre-primary education, male (number)'
            ),
            dcc.RadioItems(
                id='crossfilter-xaxis-type',
                options=[{'label': i, 'value': i}
                         for i in ['Linear', 'Log']],
                value='Linear',
                labelStyle={'display': 'inline-block'}
            )
        ],
            style={'width': '49%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='crossfilter-yaxis-column',
                options=[{'label': i, 'value': i}
                         for i in available_indicators],
                value='Population, male'
            ),
            dcc.RadioItems(
                id='crossfilter-yaxis-type',
                options=[{'label': i, 'value': i}
                         for i in ['Linear', 'Log']],
                value='Linear',
                labelStyle={'display': 'inline-block'}
            )
        ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'})
    ], style={
        'borderBottom': 'thin lightgrey solid',
        'backgroundColor': 'rgb(250, 250, 250)',
        'padding': '10px 5px'
    }),

    html.Div([
        dcc.Graph(
            id='crossfilter-indicator-scatter',
            hoverData={'points': [{'customdata': 'Albania'}]}
        )
    ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),
    html.Div([
        dcc.Graph(id='x-time-series'),
        dcc.Graph(id='y-time-series'),
    ], style={'display': 'inline-block', 'width': '49%'}),

    html.Div(dcc.Slider(
        id='crossfilter-year--slider',
        min=final_df['Years'].min(),
        max=final_df['Years'].max(),
        value=final_df['Years'].max(),
        marks={str(years): str(years)
               for years in final_df['Years'].unique()},
        step=None
    ), style={'width': '49%', 'padding': '0px 20px 20px 20px'})
])


def create_time_series(final_dff, axis_type, title):
    return {
        'data': [dict(
            x=final_dff['Years'],
            y=final_dff['Values'],
            mode='lines+markers'
        )],
        'layout': {
            'height': 225,
            'margin': {'l': 20, 'b': 30, 'r': 10, 't': 10},
            'annotations': [{
                'x': 0, 'y': 0.85, 'xanchor': 'left', 'yanchor': 'bottom',
                'xref': 'paper', 'yref': 'paper', 'showarrow': False,
                'align': 'left', 'bgcolor': 'rgba(255, 255, 255, 0.5)',
                'text': title
            }],
            'yaxis': {'type': 'linear' if axis_type == 'Linear' else 'log'},
            'xaxis': {'showgrid': False}
        }
    }


# Population variable


Introd_pop = html.Div([
    html.Div(
        [
            html.H2("Variable: Population (Total, male, female)", className="pt-5 p-2")
        ],
    )
])

##################################### POPULATION####################

# Growth population: total population VS years for the "SP.POP.TOTL"
filter_popind_final_df = final_df[(
    final_df["Indicator Code"] == "SP.POP.TOTL")]
list_of_countries_1 = filter_popind_final_df["Country Name"].unique()

graph_country_pop = html.Div([
    dcc.Checklist(
        id="list_countries",
        options=[{"label": x, "value": x}
                 for x in list_of_countries_1],
        value=list_of_countries_1[10:],
        labelStyle={'display': 'inline-block', 'padding':"5px"}
    ),
    dcc.Graph(id="pop_graph_1"),
], className="p-2")


# ranking: the highest population countries: horizontal bar ranking
# filter 2014 year:
filter_popind_final_df_2014 = filter_popind_final_df[(
    filter_popind_final_df["Years"] == 2014)]
graph_ranking_growth = dcc.Graph(
    id="ranking_graph",
    figure={
        "data": [
            go.Bar(x=filter_popind_final_df_2014["Country Name"],
                   y=filter_popind_final_df_2014["Values"],
                   marker={
                "color": "#151c97",
                "line": {
                    "color": "rgb(0, 0, 255)",
                    "width": 2,
                },
            },
                name="Population ranking 2014",
            )]
    })


# group female and male population : bar stack

# filtering 3 indicators in 2014:

filter_pop3ind_final_df = final_df[(final_df["Indicator Code"] == "SP.POP.TOTL.FE.IN") |
                                   (final_df["Indicator Code"] == "SP.POP.TOTL.MA.IN")]

filter_pop3ind_final_df = filter_pop3ind_final_df[(
    filter_pop3ind_final_df["Years"] == 2014)]

#y_value_male= filter_pop3ind_final_df[(filter_pop3ind_final_df["Indicator Code"] == "SP.POP.TOTL.FE.IN")]["Values"]


fig_graph_mf_pop = go.Figure(data=[
    go.Bar(name='Female Pop',
           x=filter_pop3ind_final_df["Country Name"].unique(
           ),
           y=filter_pop3ind_final_df[(filter_pop3ind_final_df["Indicator Code"] == "SP.POP.TOTL.FE.IN")]["Values"]),
    go.Bar(name='Male Pop',
           x=filter_pop3ind_final_df["Country Name"].unique(
           ),
           y=filter_pop3ind_final_df[(filter_pop3ind_final_df["Indicator Code"] == "SP.POP.TOTL.MA.IN")]["Values"])
])

# Change the bar mode
fig_graph_mf_pop.update_layout(barmode='stack')


graph_mf_pop = dcc.Graph(
    id="graph_malefemalepop",
    figure=fig_graph_mf_pop
)


Row_2 = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Total population', children=[
            graph_country_pop
        ]),
        dcc.Tab(label='Population by country 2014', children=[
            graph_ranking_growth
        ]),
        dcc.Tab(label='Population by gender', children=[
            graph_mf_pop
        ])
    ])
])


################GPD #########################################

Introd_gpd = html.Div([
    html.Div(
        [
            html.H2("Variable: GPD in our selected countries 2014", className="text-center p-2")
        ],
    )
])


# graph#######################################3
map_gpd = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_world_gdp_with_codes.csv')
list_code_country = final_df["Country Code"].unique()
# filtering our countries:
map_gpd = map_gpd[(map_gpd["CODE"] == list_code_country[0]) |
                  (map_gpd["CODE"] == list_code_country[1]) |
                  (map_gpd["CODE"] == list_code_country[2]) |
                  (map_gpd["CODE"] == list_code_country[3]) |
                  (map_gpd["CODE"] == list_code_country[4]) |
                  (map_gpd["CODE"] == list_code_country[5]) |
                  (map_gpd["CODE"] == list_code_country[6]) |
                  (map_gpd["CODE"] == list_code_country[7]) |
                  (map_gpd["CODE"] == list_code_country[8]) |
                  (map_gpd["CODE"] == list_code_country[9]) |
                  (map_gpd["CODE"] == list_code_country[10]) |
                  (map_gpd["CODE"] == list_code_country[11]) |
                  (map_gpd["CODE"] == list_code_country[12]) |
                  (map_gpd["CODE"] == list_code_country[13]) |
                  (map_gpd["CODE"] == list_code_country[14]) |
                  (map_gpd["CODE"] == list_code_country[15]) |
                  (map_gpd["CODE"] == list_code_country[16]) |
                  (map_gpd["CODE"] == list_code_country[17]) |
                  (map_gpd["CODE"] == list_code_country[18]) |
                  (map_gpd["CODE"] == list_code_country[19]) |
                  (map_gpd["CODE"] == list_code_country[20]) |
                  (map_gpd["CODE"] == list_code_country[21]) |
                  (map_gpd["CODE"] == list_code_country[22])]


map_graph = go.Figure(data=go.Choropleth(
    locations=map_gpd['CODE'],
    z=map_gpd['GDP (BILLIONS)'],
    text=map_gpd['COUNTRY'],
    autocolorscale=False,
    reversescale=True,
    marker_line_color='darkgray',
    marker_line_width=0.5,
    colorbar_tickprefix='$',
    colorbar_title='GDP<br>Billions US$',
))

map_graph.update_layout(
    title_text='GDP in Billions of the selected 23 Countries',
    geo=dict(
        showframe=False,
        showcoastlines=False,
        projection_type='equirectangular'
    )
)

graph_map_gpd_2014 = dcc.Graph(
    id="graph_gpd",
    figure=map_graph
)


Row_3 = html.Div([graph_map_gpd_2014])

################PISA VARIABLE #########################################

Introd_pisa = html.Div([
    html.Div(
        [
            html.H2("Variable: PISA, measure education across countries", className="p-2")
        ],
    )
])

#############################################
# graph PISA :#######################################3

# filtering average PISA indicators: LO.PISA.MAT", "LO.PISA.SCI", "LO.PISA.REA.MA"

# Growth population: total population VS years for the "SP.POP.TOTL.MA.IN"

filter_lopisa_df = final_df[(final_df["Indicator Code"] == "LO.PISA.MAT") |
                            (final_df["Indicator Code"] == "LO.PISA.SCI") |
                            (final_df["Indicator Code"] == "LO.PISA.REA.MA")]


###############Pisa AVERAGE SCORE GRAPHS###########################

available_years = [2000, 2009, 2012]

graph_lopisa = html.Div([
    dcc.Dropdown(
        id='year_selection',
        options=[{'label': i, 'value': i}
                 for i in available_years],
        value=2000
    ),
    dcc.Graph(
        id="scatter_pisa_math"
    ),
    html.P("Score:"),
    dcc.RangeSlider(
        id='range_slider',
        min=0, max=600, step=100,
        marks={290: '290', 613: '613'},
        value=[350, 450]
    ),
])


############ Checking by gender in 20012:####

# filtering 3 indicators in 2012 : "LO.PISA.MAT.FE", "LO.PISA.MAT.MA", "LO.PISA.SCI.FE", "LO.PISA.SCI.MA",
#"LO.PISA.REA.MA", "LO.PISA.REA.FE"


filter_pisagenderMAT_final_df = final_df[(final_df["Indicator Code"] == "LO.PISA.MAT.FE") |
                                         (final_df["Indicator Code"] == "LO.PISA.MAT.MA")]


filter_pisagenderSCI_final_df = final_df[(final_df["Indicator Code"] == "LO.PISA.SCI.FE") |
                                         (final_df["Indicator Code"] == "LO.PISA.SCI.MA")]

filter_pisagenderREA_final_df = final_df[(final_df["Indicator Code"] == "LO.PISA.REA.MA") |
                                         (final_df["Indicator Code"] == "LO.PISA.REA.FE")]


# graph PISA MATH by gender:

available_countries = final_df['Country Name'].unique()

graph_pisaMAT_bycountry = html.Div([
    dcc.Dropdown(
        id="select_country_pisag",
        options=[{'label': i, 'value': i}
                 for i in available_countries],
        value=available_countries[0],
        clearable=False,
    ),
    dcc.Graph(id="bar_pisa_bygender"),
])


# graph PISA REA by gender:


graph_pisaREA_bycountry = html.Div([
    dcc.Dropdown(
        id="select_country_pisagREA",
        options=[{'label': i, 'value': i}
                 for i in available_countries],
        value=available_countries[0],
        clearable=False,
    ),
    dcc.Graph(id="bar_pisa_bygenderREA"),
])


# graph PISA SCI by gender:

graph_pisaSCI_bycountry = html.Div([
    dcc.Dropdown(
        id="select_country_pisagSCI",
        options=[{'label': i, 'value': i}
                 for i in available_countries],
        value=available_countries[0],
        clearable=False,
    ),
    dcc.Graph(id="bar_pisa_bygenderSCI"),
])


Row_4 = html.Div([
    dcc.Tabs([
        dcc.Tab(label='PISA: All results by country', children=[
            graph_lopisa
        ]),
        #                         dcc.Tab(label='PISA: MAT by country', children=[
        #                             graph_pisa_bycountry
        #                         ]),
        dcc.Tab(label='PISA: REA by country', children=[
            graph_pisaREA_bycountry
        ]),
        dcc.Tab(label='PISA: SCI by country', children=[
            graph_pisaSCI_bycountry
        ])
    ])
])


#


HTML = html.Div(style={'backgroundColor': '#fdfdfd'},
                      children=[Title_eda, Introd_eda, Row_1,
                                Introd_pop, Row_2,
                                Introd_gpd, Row_3,
                                Introd_pisa, Row_4])


#


def eda_callback(app):
    @app.callback(
        dash.dependencies.Output('crossfilter-indicator-scatter', 'figure'),
        [dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
         dash.dependencies.Input('crossfilter-yaxis-column', 'value'),
         dash.dependencies.Input('crossfilter-xaxis-type', 'value'),
         dash.dependencies.Input('crossfilter-yaxis-type', 'value'),
         dash.dependencies.Input('crossfilter-year--slider', 'value')])
    def update_graph(xaxis_column_name, yaxis_column_name, xaxis_type, yaxis_type, year_value):
        global final_dff
        final_dff = final_df[final_df['Years'] == year_value]

        return {
            'data': [dict(
                x=final_dff[final_dff['Indicator Name']
                            == xaxis_column_name]['Values'],
                y=final_dff[final_dff['Indicator Name']
                            == yaxis_column_name]['Values'],
                text=final_dff[final_dff['Indicator Name']
                               == yaxis_column_name]['Country Name'],
                customdata=final_dff[final_dff['Indicator Name']
                                     == yaxis_column_name]['Country Name'],
                mode='markers',
                marker={
                    'size': 25,
                    'opacity': 0.7,
                    'color': 'orange',
                    'line': {'width': 2, 'color': 'purple'}
                }
            )],
            'layout': dict(
                xaxis={
                    'title': xaxis_column_name,
                    'type': 'linear' if xaxis_type == 'Linear' else 'log'
                },
                yaxis={
                    'title': yaxis_column_name,
                    'type': 'linear' if yaxis_type == 'Linear' else 'log'
                },
                margin={'l': 40, 'b': 30, 't': 10, 'r': 0},
                height=450,
                hovermode='closest'
            )
        }


    @app.callback(
        dash.dependencies.Output('x-time-series', 'figure'),
        [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData'),
         dash.dependencies.Input('crossfilter-xaxis-column', 'value'),
         dash.dependencies.Input('crossfilter-xaxis-type', 'value')])
    def update_y_timeseries(hoverData, xaxis_column_name, axis_type):
        global final_df
        country_name = hoverData['points'][0]['customdata']
        final_dff = final_df[final_df['Country Name'] == country_name]
        final_dff = final_dff[final_dff['Indicator Name'] == xaxis_column_name]
        title = '<b>{}</b><br>{}'.format(country_name, xaxis_column_name)
        return create_time_series(final_dff, axis_type, title)


    @app.callback(
        dash.dependencies.Output('y-time-series', 'figure'),
        [dash.dependencies.Input('crossfilter-indicator-scatter', 'hoverData'),
         dash.dependencies.Input('crossfilter-yaxis-column', 'value'),
         dash.dependencies.Input('crossfilter-yaxis-type', 'value')])
    def update_x_timeseries(hoverData, yaxis_column_name, axis_type):
        global final_df
        final_dff = final_df[final_df['Country Name']
                             == hoverData['points'][0]['customdata']]
        final_dff = final_dff[final_dff['Indicator Name'] == yaxis_column_name]
        return create_time_series(final_dff, axis_type, yaxis_column_name)


    @app.callback(
        Output("pop_graph_1", "figure"),
        [Input("list_countries", "value")])
    def update_line_chart(countries):
        global filter_popind_final_df
        select_countries = filter_popind_final_df['Country Name'].isin(countries)
        figure_gp = px.line(filter_popind_final_df[select_countries],
                            x="Years", y="Values", color='Country Name')
        return figure_gp


    @app.callback(
        Output("scatter_pisa_math", "figure"),
        [Input("range_slider", "value"),
         Input("year_selection", "value")])
    def update_scatter_chart(slider_range, selected_year):
        global filter_lopisa_df
        low, high = slider_range
        filter_lopisa_df_year = filter_lopisa_df[(
            filter_lopisa_df["Years"] == selected_year)]
        filter_range = (filter_lopisa_df_year["Values"] > low) & (
            filter_lopisa_df_year["Values"] < high)
        scatter_loaveragepisa_graph = px.scatter(
            filter_lopisa_df_year[filter_range],
            x="Country Name",
            y="Values",
            color="Indicator Code",
            size='Values',
            hover_data=['Values'])
        return scatter_loaveragepisa_graph


    @app.callback(
        Output("bar_pisa_bygender", "figure"),
        [Input("select_country_pisag", "value")])
    def update_bar_chart(country):
        global filter_pisagenderMAT_final_df
        filter_country = filter_pisagenderMAT_final_df["Country Name"] == country
        fig_graph_pisaMATbygender = px.bar(filter_pisagenderMAT_final_df[filter_country],
                                           x="Indicator Code",
                                           y="Values",
                                           color="Indicator Code",
                                           facet_col="Years",
                                           category_orders={
            "Years": [2000, 2003, 2006, 2009, 2012]})
        return fig_graph_pisaMATbygender


    @app.callback(
        Output("bar_pisa_bygenderREA", "figure"),
        [Input("select_country_pisagREA", "value")])
    def update_bar_chartREA(country):
        global filter_pisagenderREA_final_df
        filter_country = filter_pisagenderREA_final_df["Country Name"] == country
        fig_graph_pisaREAbygender = px.bar(filter_pisagenderREA_final_df[filter_country],
                                           x="Indicator Code",
                                           y="Values",
                                           color="Indicator Code",
                                           facet_col="Years",
                                           category_orders={
            "Years": [2000, 2003, 2006, 2009, 2012]})
        return fig_graph_pisaREAbygender


    @app.callback(
        Output("bar_pisa_bygenderSCI", "figure"),
        [Input("select_country_pisagSCI", "value")])
    def update_bar_chartSCI(country):
        global filter_pisagenderSCI_final_df
        filter_country = filter_pisagenderSCI_final_df["Country Name"] == country
        fig_graph_pisaSCIbygender = px.bar(filter_pisagenderSCI_final_df[filter_country],
                                           x="Indicator Code",
                                           y="Values",
                                           color="Indicator Code",
                                           facet_col="Years",
                                           category_orders={
            "Years": [2000, 2003, 2006, 2009, 2012]})
        return fig_graph_pisaSCIbygender

