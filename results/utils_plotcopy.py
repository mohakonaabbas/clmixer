"""
This module implement a interactive plot based on ploty and dash
The aim is to be able to filter easily on files
"""
from dash import Dash, html, dcc, Input, Output, callback
import pandas as pd
import plotly.express as px

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_csv('https://plotly.github.io/datasets/country_indicators.csv')

datasets=["easy","dagm","kth","magnetic","mvtec","nouvel_op"]
app.layout = html.Div([
    html.H1("Comparison of Incremental learning setups"),

    html.Div([
        html.Div([
            html.Label("Dataset",htmlFor='dataset'),
            dcc.Dropdown(
                datasets,
                'Dataset',
                id='dataset'
            )]),
        html.Div([html.Label("Backbone",htmlFor='backbone'),
                      dcc.Dropdown(
                df['Indicator Name'].unique(),
                'Life expectancy at birth, total (years)',
                id='backbone'
            )]),
        html.Div([html.Label("Model Architecture",htmlFor='archi'),
                      dcc.Dropdown(
                df['Indicator Name'].unique(),
                'Life expectancy at birth, total (years)',
                id='archi'
            )]),
        html.Div([html.Label("Knowledge Retention",htmlFor='knowretention'),
                      dcc.Dropdown(
                df['Indicator Name'].unique(),
                'Life expectancy at birth, total (years)',
                id='knowretention'
            )]),
        html.Div([html.Label("Bias Mitigation",htmlFor='bias'),
                      dcc.Dropdown(
                df['Indicator Name'].unique(),
                'Life expectancy at birth, total (years)',
                id='bias'
            )]),
        html.Div([html.Label("Uncertainty",htmlFor='uncertain'),
                      dcc.Dropdown(
                df['Indicator Name'].unique(),
                'Life expectancy at birth, total (years)',
                id='uncertain'
            )])
            ]),
        html.H2("Live plot"),

        html.Div([
        dcc.Graph(
            id='crossfilter-indicator-scatter',
            hoverData={'points': [{'customdata': 'Japan'}]}
        )
        ])

],style={'width': '99%','display': 'inline-block'})


# @callback(
#     Output('crossfilter-indicator-scatter', 'figure'),
#     Input('crossfilter-xaxis-column', 'value'),
#     Input('crossfilter-yaxis-column', 'value'),
#     Input('crossfilter-xaxis-type', 'value'),
#     Input('crossfilter-yaxis-type', 'value'),
#     Input('crossfilter-year--slider', 'value'))
# def update_graph(xaxis_column_name, yaxis_column_name,
#                  xaxis_type, yaxis_type,
#                  year_value):
#     dff = df[df['Year'] == year_value]

#     fig = px.scatter(x=dff[dff['Indicator Name'] == xaxis_column_name]['Value'],
#             y=dff[dff['Indicator Name'] == yaxis_column_name]['Value'],
#             hover_name=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name']
#             )

#     fig.update_traces(customdata=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'])

#     fig.update_xaxes(title=xaxis_column_name, type='linear' if xaxis_type == 'Linear' else 'log')

#     fig.update_yaxes(title=yaxis_column_name, type='linear' if yaxis_type == 'Linear' else 'log')

#     fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')

#     return fig


# def create_time_series(dff, axis_type, title):

#     fig = px.scatter(dff, x='Year', y='Value')

#     fig.update_traces(mode='lines+markers')

#     fig.update_xaxes(showgrid=False)

#     fig.update_yaxes(type='linear' if axis_type == 'Linear' else 'log')

#     fig.add_annotation(x=0, y=0.85, xanchor='left', yanchor='bottom',
#                        xref='paper', yref='paper', showarrow=False, align='left',
#                        text=title)

#     fig.update_layout(height=225, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})

#     return fig


# @callback(
#     Output('x-time-series', 'figure'),
#     Input('crossfilter-indicator-scatter', 'hoverData'),
#     Input('crossfilter-xaxis-column', 'value'),
#     Input('crossfilter-xaxis-type', 'value'))
# def update_y_timeseries(hoverData, xaxis_column_name, axis_type):
#     country_name = hoverData['points'][0]['customdata']
#     dff = df[df['Country Name'] == country_name]
#     dff = dff[dff['Indicator Name'] == xaxis_column_name]
#     title = '<b>{}</b><br>{}'.format(country_name, xaxis_column_name)
#     return create_time_series(dff, axis_type, title)


# @callback(
#     Output('y-time-series', 'figure'),
#     Input('crossfilter-indicator-scatter', 'hoverData'),
#     Input('crossfilter-yaxis-column', 'value'),
#     Input('crossfilter-yaxis-type', 'value'))
# def update_x_timeseries(hoverData, yaxis_column_name, axis_type):
#     dff = df[df['Country Name'] == hoverData['points'][0]['customdata']]
#     dff = dff[dff['Indicator Name'] == yaxis_column_name]
#     return create_time_series(dff, axis_type, yaxis_column_name)


if __name__ == '__main__':
    app.run(debug=True)
