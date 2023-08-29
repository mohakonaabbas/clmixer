"""
This module implement a interactive plot based on ploty and dash
The aim is to be able to filter easily on files
"""
from dash import Dash, html, dcc, Input, Output, callback
import pandas as pd
import plotly.express as px
import plot_api
import plotly.graph_objects as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

df = plot_api.getAllValidExperiments(databaseName="experiments_representation_fixed")
labels=plot_api.getUniqueValues(df)

metrics = ["acc",'mica']
app.layout = html.Div([
    html.H1("Comparison of Incremental learning setups",style={"text-align":"center"}),

    html.Div([
        html.Div([
            html.Label("Dataset",htmlFor='dataset'),
            dcc.Checklist(
                options=labels["dataset"],value=labels["dataset"],inline=True,id='dataset')]),
            html.Div([html.Label("Scenario",htmlFor='scenario'),
                      dcc.Checklist(
                options=labels["scenario"],value=labels["scenario"],inline=True, id='scenario'
            )]),
        html.Div([html.Label("Backbone",htmlFor='backbone'),
                      dcc.Checklist(
                options=labels["backbone"],value=labels["backbone"],inline=True,
                id='backbone'
            )]),
        html.Div([html.Label("Model Architecture",htmlFor='archi'),
                      dcc.Checklist(
                options=labels["architecture"],value=labels["architecture"],inline=True,
                id='archi'
            )]),
        html.Div([html.Label("Knowledge Retention",htmlFor='knowretention'),
                      dcc.Checklist(
                options=labels["retention"],value=labels["retention"],inline=True,
                id='knowretention'
            )]),
        html.Div([html.Label("Bias Mitigation",htmlFor='bias'),
                      dcc.Checklist(
                options=labels["bias"],value=labels["bias"],inline=True,
                id='bias'
            )]),
        html.Div([html.Label("Uncertainty",htmlFor='uncertain'),
                      dcc.Checklist(
                options=labels["uncertainty"], value=labels["uncertainty"],inline=True,
                id='uncertain'
            )]),
                    html.Div([html.Label("Metrics",htmlFor='metric'),
                      dcc.Checklist(
                options=metrics, value=metrics,inline=True,
                id='metric'
            )])
            ]),
        html.H2("Live filtered plot"),

        html.Div([
        dcc.Graph(
            id='graph'
            # hoverData={'points': [{'customdata': 'Japan'}]}
        )
        ])

])


@callback(
    Output('graph', 'figure'),
    Input('dataset', 'value'),
    Input('scenario', 'value'),
    Input('backbone', 'value'),
    Input('archi', 'value'),
    Input('knowretention', 'value'),
    Input('bias', 'value'),
    Input('uncertain', 'value'),
    Input('metric', 'value'))
def update_graph(dataset,
                 scenario,
                 backbone, 
                 archi,
                 knowretention,
                 bias,
                 uncertain,metrics):
    
    filter_dict={}
    filter_dict["dataset"]=dataset
    filter_dict["scenario"]=scenario
    filter_dict["backbone"]=backbone
    filter_dict["architecture"]=archi
    filter_dict["retention"]=knowretention
    filter_dict["bias"]=bias
    filter_dict["uncertainty"]=uncertain

    print(filter_dict)
    plots=[]
    
    filtered_df=plot_api.filterValidExperiments(df,filter_dict)
    for metric in metrics:
        to_plot=plot_api.formatValuesToPlottyLines(filtered_df, metric=metric)

        plots+=list(to_plot.to_dict().values())

    # Loop over all the plots
    fig=go.Figure()
    for i in range(len(plots)):
        fig.add_trace(go.Scatter(x=plots[i]["x"],y=plots[i]["y"],mode=plots[i]["mode"],name=plots[i]["name"]))










    # fig = px.scatter(x=dff[dff['Indicator Name'] == xaxis_column_name]['Value'],
    #         y=dff[dff['Indicator Name'] == yaxis_column_name]['Value'],
    #         hover_name=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name']
    #         )

    # fig.update_traces(customdata=dff[dff['Indicator Name'] == yaxis_column_name]['Country Name'])

    # fig.update_xaxes(title=xaxis_column_name, type='linear' if xaxis_type == 'Linear' else 'log')

    # fig.update_yaxes(title=yaxis_column_name, type='linear' if yaxis_type == 'Linear' else 'log')

    # fig.update_layout(margin={'l': 40, 'b': 40, 't': 10, 'r': 0}, hovermode='closest')

    return fig


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
