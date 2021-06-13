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
import plotly.express as px

# import data frame
df = pd.read_csv("edstats-csv-zip-32-mb-/EdStatsData.csv")

# Creating our tables:
# name of the file: EdStatsData.csv
file_name = "EdStatsData.csv"

# file info:
num_countries = len(df['Country Code'].unique())
num_indicators = len(df['Indicator Code'].unique())


# summary 1
# initialize list
summary_table_1 = [[file_name, df.shape[1], df.shape[0]]]
# Create the pandas DataFrame
summary_table_1 = pd.DataFrame(summary_table_1, columns=[
                               'File name', 'Num of Columns', "Num of rows"])
# summary table 2
# initialize list
summary_table_2 = [["Countries", num_countries], [
    "Indicators", num_indicators], ["Years", "From 1970"]]
# Create the pandas DataFrame
summary_table_2 = pd.DataFrame(summary_table_2, columns=[
                               'Variables', 'Observations'])


# Filtering year
main_cols = ['Country Code', 'Country Name', 'Indicator Code',
             'Indicator Name'] + [str(i) for i in range(1990, 2015)]
df = df[main_cols]


# summary table 3
# initialize list of lists
summary_table_3 = [["Years", "From 1990 to 2015"]]
# Create the pandas DataFrame
summary_table_3 = pd.DataFrame(summary_table_3, columns=[
                               'Variables', 'Observations'])


# 

summary_tables = html.Div([
    html.Div([
        html.H1('Our original data', className="h1 text-center font-weight-bold"),
        dash_table.DataTable(
            data=summary_table_1.to_dict('records'),
            columns=[{"name": i, "id": i} for i in summary_table_1.columns],
            id='Summary table 1',
            style_cell_conditional=[{'textAlign': 'center'}],
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
        )
    ], className="col-12 p-5"),  # 'display': 'inline-block'
    html.Div([
        html.H2('Into the variables',  className="h2 text-center pt-4"),
        dash_table.DataTable(
            data=summary_table_2.to_dict('records'),
            columns=[{"name": i, "id": i} for i in summary_table_2.columns],
            id='Summary table 2',
            style_cell_conditional=[{'textAlign': 'center'}],
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
        )
    ], className="col-6"),
    html.Div([
        html.H2('Filtering the years',  className="h2 text-center pt-4"),
        dash_table.DataTable(
            data=summary_table_3.to_dict('records'),
            columns=[{"name": i, "id": i} for i in summary_table_3.columns],
            id='Summary table 3',
            style_cell_conditional=[{'textAlign': 'center'}],
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
        )
    ],className="col-6")
], className="row")

# Graphs

Header = html.Div([
    html.Br([]),
    html.H1('Our data frame: Select Country & Indicator', className="h1 text-center font-weight-bold")])


General_graph_1 = html.Div([
    html.H2('Indicators through the years'),
    dcc.Dropdown(  # input1
        id='countryname-dropdown',
        options=[{'label': i, 'value': i}
                 for i in df['Country Name'].unique()],
        value='Sweden'
    ),
    dcc.Dropdown(
        id='indicator-dropdown',  # input2
        options=[{'label': i, 'value': i}
                 for i in df['Indicator Name'].unique()],
        value='Adjusted net enrolment rate, primary, both sexes (%)'
    ),
    dcc.Graph(id='graph_total_growth')
], style={'width': '100%'})


General_graph_2 = html.Div([
    html.H2("Percentage of Null values in the indicator", className="text-center font-weight-bold pt-4"),
    html.H5('  '),
    html.H5('Considering the selected country and indicator, we can observe how many empty values we have.'),
    dcc.Graph(id='pie-chart')
], style={'width': '100%'})



HTML = html.Div(style={'backgroundColor': '#fdfdfd'}, children=[
                      summary_tables, Header, General_graph_1, General_graph_2])


###############

def summary_callback(app):

    @app.callback(
        [
            Output('graph_total_growth', 'figure'),
            Output('pie-chart', 'figure')
        ],
        [
            Input('countryname-dropdown', 'value'),
            Input('indicator-dropdown', 'value')
        ]
    )
    def update_chartSummary(selected_dropdown_value1, selected_dropdown_value2):
        global df
        filtered_data = df[df["Country Name"] == selected_dropdown_value1]
        filtered_data = filtered_data[filtered_data["Indicator Name"]
                                      == selected_dropdown_value2]
        filtered_data = filtered_data.melt(
            id_vars=["Country Name", "Indicator Name"],
            value_vars=['1990', '1991', '1992', '1993', '1994', '1995',
                        '1996', '1997', '1998', '1999', '2000', '2001',
                        '2001', '2001', '2002', '2003', '2004', '2005',
                        '2006', '2007', '2008', '2009', '2010', '2011',
                        '2012', '2013', '2014'],
            var_name='Years', value_name='Values')
        figure1 = px.scatter(filtered_data, x='Years', y='Values',
                             title=f'Growth of {selected_dropdown_value1} indicator in {selected_dropdown_value2} ')

        pie_data = [["Empty Values", filtered_data.isnull().sum().sum()],
                    ["Complete data", len(filtered_data["Values"])-filtered_data.isnull().sum().sum()]]
        pie_data = pd.DataFrame(pie_data, columns=['Names', 'Values'])
        figure2 = px.pie(pie_data, values="Values", names="Names",
                         title=f'Rate of empty values in the indicator {selected_dropdown_value2}')

        return figure1, figure2
