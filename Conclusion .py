import dash_html_components as html
import dash
import plotly.graph_objs as go



Title_conclusion = html.Div([
    html.H1('Conclusion'),
    html.Br([]), html.P(
        "\
                                        Our main conclusions are: ",
        style={"color": "#000000"},
        className="row",
    ),
])

# title conclusion 


text_conclusion = html.Div([
    html.Div(
        [
            html.H2("Questions"),
            html.Br([]),
            html.H3(
                "\
                Which countries have the best and the worst performance in terms of education?",
                style={"color": "#104cce"}
            ),
            
           html.P(
                "\
                The best were: China, Singapore, Hong Kong, Korea Rep, Macao SAR, Japan; \
                The worst were: Peru, Indonesia, Qatar, Colombia, Jordan, Tunisia.",
                style={"color": "#000000"}
            ),
            
            html.H3(
                "\
                Which countries have growth more in the last 10 years in terms of education? ",
                style={"color": "#104cce"}
            ),
            
           html.P(
                "\
                The highest growth: Peru, Brazil, Poland, Chile, Luxembourg, Israel;\
                The one who dicrease in the growth rate: New Zealand, UK, Sweden, Australia, France, Iceland.",
                style={"color": "#000000"}
            ),
            
            html.H3(
                "\
                Between the 3 regression models applied for the prediction, which one is the best?",
                style={"color": "#104cce"}
            ),
            
           html.P(
                "\
                Decision Trees has the highest R2 ",
                style={"color": "#000000"}
            ),
        ], style={'width': '100%', 'display': 'inline-block'}
    )
])


######################

HTML =  html.Div(style={'backgroundColor': '#fdfdfd'},
                      children=[Title_conclusion, text_conclusion])
