import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import numpy as np

# Provided dictionary
data = {
    'x': [0, 1, 2, 3, 4],
    'y': [0.89999999999997, 0.8918918918918678, 0.686746987951799, 0.923076923076913, 0.8809523809523705],
    'mode': 'lines+markers',
    'name': 'kth_induscil_mlp_dinov2vits14_CrossEntropy_KnowledgeDistillation_Finetune_None'
}

# Generate a distribution of values around the provided y values
np.random.seed(42)  # For reproducibility
x_values = data['x']
y_values = data['y']
data_with_distribution = {x: np.random.normal(loc=y, scale=0.05, size=50) for x, y in zip(x_values, y_values)}

# Calculate the mean for each category
means = {x: np.mean(values) for x, values in data_with_distribution.items()}

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Box Plot with Means Example"),
    dcc.Graph(
        id='box-plot',
        figure={
            'data': [
                go.Box(
                    x=[x]*len(data_with_distribution[x]),
                    y=data_with_distribution[x],
                    name=f'Category {x}',
                    marker_color='blue',
                    showlegend=False
                ) for x in x_values
            ] + [
                go.Scatter(
                    x=x_values,
                    y=[means[x] for x in x_values],
                    mode=data['mode'],
                    marker=dict(color='black', size=10),
                    name='Mean'
                )
            ],
            'layout': go.Layout(
                title='Box Plot with Means',
                xaxis=dict(title='Categories'),
                yaxis=dict(title='Values'),
                boxmode='overlay'  # Overlay the box plots on the same x-axis locations
            )
        }
    )
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
