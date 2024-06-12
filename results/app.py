"""
This module implement a interactive plot based on ploty and dash
The aim is to be able to filter easily on files
"""
from dash import Dash, html, dcc, Input, Output, callback
import pandas as pd
import plotly.express as px
import plot_api
import plotly.graph_objects as go
import random
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

# df1 = plot_api.getAllValidExperiments(databaseName="representation_fixed")
df = plot_api.getAllValidExperiments(databaseName="Frozen")
# df = pd.concat([df1,df2]).reset_index(drop=True)
labels=plot_api.getUniqueValues(df)

metrics = ["acc",'mica']

# Function to generate a random color
def random_color():
    return f'#{random.randint(0, 0xFFFFFF):06x}'

app.layout = html.Div([
    html.H1("Comparison of Incremental learning setups",style={"text-align":"center"}),
    html.Div([
    html.Div([html.H2("Parameters"),

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
                    html.Div([html.Label("Incorporation",htmlFor='incorp'),
                      dcc.Checklist(
                options=labels["incorporation"], value=labels["incorporation"],inline=True,
                id='incorp'
            )])
            ])],className="parameter_div"),
    html.Div([html.H2("Live filtered plot"),
              html.Div([dcc.Graph(id='graph_acc')]),
              html.Div([dcc.Graph(id='graph_mica')])],className="graph_div")],className="app_div")
])


@callback(
    Output('graph_acc', 'figure'),
    Input('dataset', 'value'),
    Input('scenario', 'value'),
    Input('backbone', 'value'),
    Input('archi', 'value'),
    Input('knowretention', 'value'),
    Input('bias', 'value'),
    Input('uncertain', 'value'),
    Input('incorp', 'value'))
def update_graph(dataset,
                 scenario,
                 backbone, 
                 archi,
                 knowretention,
                 bias,
                 uncertain,incorporation):
    
    filter_dict={}
    filter_dict["dataset"]=dataset
    filter_dict["scenario"]=scenario
    filter_dict["backbone"]=backbone
    filter_dict["architecture"]=archi
    filter_dict["retention"]=knowretention
    filter_dict["bias"]=bias
    filter_dict["uncertainty"]=uncertain
    filter_dict["incorporation"]=incorporation

    print(filter_dict)
    plots=[]
    
    filtered_df=plot_api.filterValidExperiments(df,filter_dict)
    to_plot=plot_api.formatValuesToPlottyLines(filtered_df, metric="acc")
    plots+=list(to_plot.to_dict().values())

    # Loop over all the plots
    fig=go.Figure(layout={"title":"Acc"})
    for i in range(len(plots)):

        color = random_color() 
        
        scatter_data = plots[i][0]
        box_data = plots[i][1]
        fig.add_trace(go.Scatter(x=scatter_data["x"],
                                 y=scatter_data["y"],
                                 mode=scatter_data["mode"],
                                 name=scatter_data["name"],
                                marker=dict(color=color, size=10)))
        
        
        for k in range(len(box_data)):
            datum=box_data[k]
            fig.add_trace(go.Box(
                    x=datum["x"],
                    y=datum["y"],
                    marker_color=color,
                    showlegend=False
                ) )
    return fig

@callback(
    Output('graph_mica', 'figure'),
    Input('dataset', 'value'),
    Input('scenario', 'value'),
    Input('backbone', 'value'),
    Input('archi', 'value'),
    Input('knowretention', 'value'),
    Input('bias', 'value'),
    Input('uncertain', 'value'),
    Input('incorp', 'value'))
def update_graph_2(dataset,
                 scenario,
                 backbone, 
                 archi,
                 knowretention,
                 bias,
                 uncertain,incorporation):
    
    filter_dict={}
    filter_dict["dataset"]=dataset
    filter_dict["scenario"]=scenario
    filter_dict["backbone"]=backbone
    filter_dict["architecture"]=archi
    filter_dict["retention"]=knowretention
    filter_dict["bias"]=bias
    filter_dict["uncertainty"]=uncertain
    filter_dict["incorporation"]=incorporation

    print(filter_dict)
    plots=[]
    
    filtered_df=plot_api.filterValidExperiments(df,filter_dict)
    to_plot=plot_api.formatValuesToPlottyLines(filtered_df, metric="mica")

    plots+=list(to_plot.to_dict().values())

    # Loop over all the plots
    fig=go.Figure(layout={"title":"Mica"})
    for i in range(len(plots)):
        scatter_data = plots[i][0]
        box_data = plots[i][1]
        fig.add_trace(go.Scatter(x=scatter_data["x"],
                                 y=scatter_data["y"],
                                 mode=scatter_data["mode"],
                                 name=scatter_data["name"]))
        
        # for k in range(len(box_data)):
        #     datum=box_data[k]
        #     fig.add_trace(go.Box(
        #             x=datum["x"],
        #             y=datum["y"],
        #             marker_color='blue'
        #         ) )

    return fig

if __name__ == '__main__':
    app.run(debug=True)
