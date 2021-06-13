from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn import linear_model
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
from Preprocessing import final_df
from warnings import filterwarnings
filterwarnings("ignore")


#


final_df = final_df[final_df.columns[0:29]]

# pivot from wide to long:

final_df = final_df.melt(id_vars=["Country Name", "Country Code", "Indicator Name", "Indicator Code"], value_vars=['1990', '1991', '1992', '1993', '1994', '1995',
                                                                                                                   '1996', '1997', '1998', '1999', '2000', '2001',
                                                                                                                   '2001', '2001', '2002', '2003', '2004', '2005',
                                                                                                                   '2006', '2007', '2008', '2009', '2010', '2011',
                                                                                                                   '2012', '2013', '2014'], var_name='Years', value_name='Values')


# Passing values to integer:

final_df['Years'] = final_df['Years'].astype(int)

# Delete the NAN Values
final_df = final_df.dropna()

####### pivot tables################ 

#select the indicators to forecast:

ind_forecast = ["LO.PISA.MAT",# PISA: Mean performance on the mathematics scale.
                 "SP.POP.TOTL.MA.IN", "SP.POP.TOTL.FE.IN",   #Population, male, female
                 "SP.PRM.TOTL.FE.IN", "SP.PRE.TOTL.MA.IN",  # Pop of the official age for primar education male, female
                 "NY.GDP.PCAP.PP.KD" # GDP per capita, PPP
                        ] # maybe we can include "LO.PISA.MAT.FE", "LO.PISA.MAT.MA" later 

index= ["Country Name", "Years"]
columns = ["Indicator Code"]


#filter the data frame

final_df = final_df[final_df['Indicator Code'].isin(ind_forecast)]

########################for pivot_table

final_df= final_df[["Country Name","Indicator Code", "Years", "Values"]]
final_df= pd.pivot_table(data=final_df, index=["Country Name", "Years"] , columns ="Indicator Code")

###############################################

final_df = final_df.dropna()

#select the values for filtering index :
final_df = final_df['Values']
final_df.index.name = None   # delete index name 
final_df= final_df.reset_index() # reset index 

###### reorganize the data frame

#order of columns: 

arrange_final= ["LO.PISA.MAT",# PISA: Mean performance on the mathematics scale.
                        "Country Name", "Years",
                        "SP.POP.TOTL.MA.IN", "SP.POP.TOTL.FE.IN",   #Population, male, female
                        "SP.PRM.TOTL.FE.IN", "SP.PRE.TOTL.MA.IN",  # Population of the official age for primary education male, female
                        "NY.GDP.PCAP.PP.KD" # GDP per capita, PPP
                        ] # maybe we can include "LO.PISA.MAT.FE", "LO.PISA.MAT.MA" later 

#final order of the data frame:
final_df = final_df[arrange_final]

####################################################


#


Title_model = html.Div([
    html.H1('Model', className="display-4 font-weight-bolder text-center",),
    html.Br([]), 
    html.P(
        "We are going to apply 2 models: Linear Regression, Decision Trees",
       className="h5"
    ),
])

######################## DUMMY VARIABLES #############################################
Introd_dummyvariable = html.Div(
    [
        html.Div(
            [
                html.H2("Create Dummy Variables", className="font-weight-bold pt-5"),
                html.Br([]),
            ],
            className="product",
        )
    ])

################# Overall ################################################
#final_df = final_df[["Country Name", "Indicator Name", "Years", "Values"]]

#create dummy variables to countries: 
final_df_wdummyv = pd.get_dummies(
    final_df, columns=["Country Name"], drop_first=True) # "Years","Indicator Name"


###### Summary table with new dummy variables #########################################################

# initialize list
summary_table_dummy1 = [["Our filter data set",
                         final_df_wdummyv.shape[1], final_df_wdummyv.shape[0]]]
# Create the pandas DataFrame
summary_table_dummy1 = pd.DataFrame(summary_table_dummy1, columns=[
                                    'Name', 'Num of Columns', "Num of rows"])


####summary table 2: filter##########################
# initialize list

dummy_countries = final_df_wdummyv.columns[7:27]
indep_variable = ["Years",
                    "SP.POP.TOTL.MA.IN", "SP.POP.TOTL.FE.IN",   #Population, male, female
                    "SP.PRM.TOTL.FE.IN", "SP.PRE.TOTL.MA.IN",  # Population of the official age for primary education male, female
                    "NY.GDP.PCAP.PP.KD"]
dep_variable = ["LO.PISA.MAT"]

# initialize list 
summary_table_dummy2 = [["Dummy countries",dummy_countries, len(dummy_countries)],
                          ["Independet variables",indep_variable, len(indep_variable)],
                          ["Dependet variables",dep_variable, len(dep_variable)]]

# Create the pandas DataFrame
summary_table_dummy2 = pd.DataFrame(summary_table_dummy2, columns = ['Variable', "List", 'Quantity'])



############generate the summary tables####################

summary_tables_dummy = html.Div([
    html.Div([
                                html.H4('Our data with dummy variables', className="h4"),
                                dash_table.DataTable(
                                    data=summary_table_dummy1.to_dict(
                                        'records'),
                                    columns=[{"name": i, "id": i}
                                             for i in summary_table_dummy1.columns],
                                    id='summary_dummy1',
                                    style_cell_conditional=[
                                        {'textAlign': 'center'}],
                                    style_header={
                                        'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
                                )
                                ], className="col-12 py-4"),  # 'display': 'inline-block'
    html.Div([
        html.H4('Our dummy Variables', className="h4"),
        dash_table.DataTable(
            data=summary_table_dummy2.to_dict(
                'records'),
            columns=[{"name": i, "id": i}
                     for i in summary_table_dummy2.columns],
            id='Summary_dummy2',
            style_cell_conditional=[
                {'textAlign': 'center'}],
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'}
        )
    ], className="col-12")
], className="row")


######################## INTRO TITLE Split of the data: training and test #############################################
Introd_split = html.Div([
    html.Div(
        [
            html.Br([]),
            html.H4("Training and Testing Data", className="h4"),
            html.Br([]),
        ],
        className="product",
    )
])

################################## training and test pie charts###############

# Importing the dataset
y= final_df_wdummyv[["LO.PISA.MAT"]]  #dependet variable
x= final_df_wdummyv.drop('LO.PISA.MAT',axis=1) #independent variables

# split the training and testing
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=10)


split = ["Train ", "Test "]
percent = [len(x_train)/(len(x_train) + len(x_test)),
           len(x_test)/(len(x_train) + len(x_test))]


pie_traintest = dcc.Graph(
    id="topoverall6",
    figure=go.Figure(data={
        "values": percent,
        "labels": split,
        "domain": {"column": 0},
        "name": "split",
        "hoverinfo": "percent",
        "hole": .4,
        "type": "pie"
    },
        layout=go.Layout(
        {
            "title": "Train VS Test",
            "annotations": [
                {
                    "font": {
                        "size": 20
                    },
                    "showarrow": False,
                    "text": "Split",
                    "x": 0.20,
                    "y": 0.5
                }
            ]
        }
    ))
)


######################## Intro models:###################

Introd_selectedmodel = html.Div([
                                html.Div(
                                    [
                                        html.H2("Models to apply:Lr, Dt, Rf", className="font-weight-bold pt-5"),
                                        html.Br([]),
                                    ],
                                    className="product",
                                )
                                ])


###linear regression###########################################

# call multivariant regression model:
linearmodel = linear_model.LinearRegression()
# fit the model with the data
linearmodel.fit(x_train, y_train)
# after the model have learned, move on ty predict with the testing data:
y_predicted_lr = linearmodel.predict(x_test)

#data frame for create a list from the y_predicted [[]]--> []

y_predicted_lr2 = []
for l in y_predicted_lr:
    y_predicted_lr2.extend(l)

# data frame for compare the outputs in the graphs
list_pred= y_test["LO.PISA.MAT"]
comp_lr = pd.DataFrame({'Test': list_pred.tolist(), 
                        'Prediction': list(y_predicted_lr2)}, 
                       columns=['Test', 'Prediction'])


    
scatter_lr = dcc.Graph(
    id="lrgraph",
    figure=go.Figure(data=go.Scatter(
        x=comp_lr["Test"], y=comp_lr["Prediction"], mode='markers'))
)

############ Decision trees###############################

# create a regressor object
decision_trees_model = DecisionTreeRegressor(random_state=0)

# fit the regressor with X and Y data
decision_trees_model.fit(x_train, y_train)

# predicting a new value
# test the output by changing values
y_predicted_dt = decision_trees_model.predict(x_test)

# data frame for compare the outputs in the graphs
list_pred= y_test["LO.PISA.MAT"]
comp_dt = pd.DataFrame({'Test': list_pred.tolist(), 'Prediction': list(y_predicted_dt)}, columns=['Test', 'Prediction'])

scatter_dt = dcc.Graph(
    id="dtgraph",
    figure=go.Figure(data=go.Scatter(
        x=comp_dt["Test"], y=comp_dt["Prediction"], mode='markers'))
)


##############Random forest#######################################

# create regressor object
random_forest_model = RandomForestRegressor(random_state=0)

# fit the regressor with x and y data
random_forest_model.fit(x_train, y_train)

# predicting a new value
# test the output by changing values, like 3750
y_predicted_rf = random_forest_model.predict(x_test)

# data frame for compare the outputs in the graphs
list_pred= y_test["LO.PISA.MAT"]
comp_rf = pd.DataFrame({'Test': list_pred.tolist(), 'Prediction': list(y_predicted_rf)}, columns=['Test', 'Prediction'])

scatter_rf = dcc.Graph(
    id="rfgraph",
    figure=go.Figure(data=go.Scatter(
        x= comp_rf['Test'], y=comp_rf['Prediction'], mode='markers'))
)


Row_1 = html.Div([
    dcc.Tabs([
        dcc.Tab(label='Linear Regression', children=[
            scatter_lr
        ]),
        dcc.Tab(label='Decision Trees', children=[
            scatter_dt
        ]),
        dcc.Tab(label='Random Forest', children=[
            scatter_rf
        ])
    ])
])

# Evaluate the models


r2_linear_model = r2_score(y_test, y_predicted_lr)
r2_dt_model = r2_score(y_test, y_predicted_dt)
r2_rf_model = r2_score(y_test, y_predicted_rf)


####summary table 2: r2##########################
# initialize list
summary_table_evaluation = [["Linear Regression", round(r2_linear_model, 3)],
                            ["Decision Trees", round(r2_dt_model, 3)],
                            ["Random Forest", round(r2_rf_model, 3)]]
# Create the pandas DataFrame
summary_table_ev = pd.DataFrame(
    summary_table_evaluation, columns=['Model', "R2"])


##################################

Evaluation = html.Div([
    html.Div(
        [
            html.H2("Evaluating the models", className="font-weight-bold pt-5"),
            html.Br([]),
            html.P(
                "Since we are doing a regression models, let's mesure the R2.",
                className="text-dark"
            ),
        ], style={'width': '100%', 'display': 'inline-block'}
    ),
    html.Div([
        # column 2 table
        dash_table.DataTable(
            data=summary_table_ev.to_dict('records'),
            columns=[{"name": i, "id": i}
                     for i in summary_table_ev.columns],
            id='summary_evaluation',
            style_cell_conditional=[{'textAlign': 'center'}],
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
            style_data_conditional=[ {'if': {'filter_query': '{{R2}} = {}'.format(summary_table_ev['R2'].max()),},
                                      'backgroundColor': '#FF4136',
                                      'color': 'white'}, ]
        )
    ], style={'width': '100%', 'display': 'inline-block'}
    ),
    html.Div(
        [
            html.P(
                """The choosen model is Decision Trees.""",
                className="row bg-dark text-warning p-2 rounded",
            ),
        ],
        className="py-3 h5 ",
    )
])


################### Predictions ####################################

##################################

Intro_pred_10years = html.Div([
    html.Div(
        [
            html.H2("How it will go the next 10 years??", className="font-weight-bold pt-5"),
            html.Br([]),
            html.P(
                "Prediction of the next 10 years (Years VS PISA.MAT).",
                style={"color": "#000000"}
            ),
        ], style={'width': '100%', 'display': 'inline-block'}
    )
])


# data arrange for the graph:


# for being able to predict we need to modify the year indicator in the x_Test data frame
# also we keep all the variables stack in 2014

predicted_years = np.arange(2015, 2028, 3) #every 3 years the exam is taken 
predicted_years= np.repeat(predicted_years,repeats=1)  #here i have 5 times the array created before

x_test_forprediction= x_test[ (x_test["Years"] == 2012) ] #filtering the 2012 because it the last year the exam 

x_test_forprediction['Years']= predicted_years
x_test_forprediction = pd.concat([x_test_forprediction]*4)
x_test_forprediction = x_test_forprediction[0:18]

# we call the decision trees training model:

y_predicted_dt_next10years = decision_trees_model.predict(x_test_forprediction)

##############graph prediction####################################################

# Create traces
prediction_graph = go.Figure()
prediction_graph.add_trace(go.Scatter(x=x_test_forprediction["Years"],
                                      y=y_predicted_dt_next10years,
                                      mode='markers',
                                      name='Prediction'))
prediction_graph.add_trace(go.Scatter(x=x_test["Years"],
                                      y=y_predicted_rf,
                                      mode='markers',
                                      name='Test data'))


line_graph_testandprediction = dcc.Graph(
    id="testandpred",
    figure=prediction_graph)


#########

HTML =  html.Div(style={'backgroundColor': '#fdfdfd'},
                      children=[Title_model, Introd_dummyvariable, summary_tables_dummy,
                                Introd_split, pie_traintest,
                                Introd_selectedmodel, Row_1,
                                Evaluation,
                                Intro_pred_10years,
                                line_graph_testandprediction])