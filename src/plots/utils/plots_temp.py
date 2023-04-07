from dash import Dash, dash_table, dcc, html, State
from dash.dependencies import Input, Output
import pandas as pd
from dash import Dash, dash_table, dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import data_utils
import plotly.express as px
import mlflow_util
import sys
import os
import dash_bootstrap_components as dbc
from plotly.subplots import make_subplots
import plotly.graph_objects as go

folder_name = r"c:\dev\rl-fn\mlruns_sc\mlruns"
if (len(sys.argv)) > 1:
    folder_name = sys.argv[1]
pkl_name = os.path.join(folder_name, "local.pkl")

_, df = data_utils.get_all_data_dash(pkl_name)
df_selected = df
print(df.columns)

app = Dash(__name__)

app.layout = html.Div([
    dbc.Row(
        [html.Div(id='record-statistics'),
         dcc.Dropdown(
            id="filter_dropdown",
            options=[{"label": st, "value": st} for st in df.name.unique()],
            placeholder="-Select an experiment-",
            multi=True,
            value=df.name.unique(),
        ),
            html.Div(id='table-container'),
            dash_table.DataTable(
            id='datatable-interactivity',
            columns=[{"name": i, "id": i} for i in df.columns],
            data=df.to_dict('records'),
            editable=False,
            filter_action="native",
            sort_action="native",
            sort_mode="multi",
            column_selectable="single",
            row_selectable="single",
            row_deletable=True,
            selected_columns=[],
            selected_rows=[],
            page_action="native",
            page_current=0,
            export_format="csv",
            page_size=5,

            style_data={
                'color': 'black',
                'backgroundColor': 'white'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(220, 220, 220)',
                }
            ],
            style_header={
                'backgroundColor': 'rgb(210, 210, 210)',
                'color': 'black',
                'fontWeight': 'bold'
            },
            style_cell_conditional=[
                {'if': {'column_id': c},
                 'display': 'None', } for c in ['exp_id', 'run_id']]


        )
            

        ]),
    dbc.Row([
        dbc.Col([
            dcc.Checklist(
                options=[{'label':x,'value':x} for x in df.columns],
                id='df-cols'
            )]

        ),
        dbc.Col(
            [
                dcc.Checklist(
                    options=[{'label':x,'value':x} for x in df.columns],
                    id='df-pivot'
                )]

        ),

        dbc.Col([
            dcc.Dropdown(
                options=[{'label':x,'value':x} for x in ['Heatmap', 'BoxPlots', 'Histograms']],
                id='df-figure'
            )
        ])
    ]),
    dbc.Row([
        dcc.Graph(id="graph-plot"),
        dcc.Graph(id="statistics-plot")
    ]
    )
],style={"font-family": "Arial", "font-size": "0.9em", "text-align": "center"})
@app.callback(
    Output("datatable-interactivity", "data"), 
    Output("record-statistics","children"),
    Input("filter_dropdown", "value"),
    Input('datatable-interactivity', 'derived_virtual_data'),
    prevent_initial_call=True
)
def display_table(name,virtual_data):
    global df_selected
    dff = df[df.name.isin(name)]
    df1 = pd.DataFrame.from_dict(virtual_data)
    if df1 is None or df1.shape[0]==0:
        df_selected = df1
        return dff.to_dict("records"),html.H3('Number of selected records {}'.format(0))
    df1=df1[df1["name"].isin(name)]
    rlen = df1.shape[0]
    df_selected =df1
    return dff.to_dict("records"),html.H3('Number of selected records {}'.format(rlen))


@app.callback(
    Output('graph-plot', 'figure'),
    Output('statistics-plot','figure'),
    Input('df-cols','value'),
    Input('df-pivot','value'),
    Input('df-figure','value'),
    prevent_initial_call=True
)
def draw_graph(cv,pv,fv):
    global df_selected
    if df_selected is None:
        return {},{}
    if cv == None or len(cv)< 1:
        return {},{}
    df1 = df_selected
    r=1
    c=1
    row_t = ['']
    col_t = ['']
    if pv and len(pv)>=1:
        c=len(df1[pv[0]].unique())
        col_t= df1[pv[0]].unique()
    if pv and len(pv)==2:
        r =len(df1[pv[1]].unique())
        row_t =df1[pv[1]].unique()
    fig= make_subplots(r,c)
    df2 = df1

    for r1 in range(r):
        row_s =''
        if pv and len(pv) ==2:
            df2 = df2[df2[pv[1]] == row_t[r1]]
            row_s=row_t[r1]
        for c1 in range(c):
            df3 = df2
            col_s = ''
            if pv and len(pv)>=1:
                df3 =  df2[df2[pv[0]] == col_t[c1]]
                col_s = col_t[c1]
                if fv=='BoxPlots':
                    fig.add_trace(
                        go.Box(y=df3[cv[0]],
                        name='{}-{}'.format(row_s,col_s)),
                        row=r1+1, col=c1+1
                        )
                elif fv=='Histograms':
                    if cv and len(cv)==1:
                        fig.add_trace(go.Histogram(x=df3[cv[0]],name='{}-{}'.format(row_s,col_s)),row=r1+1, col=c1+1)
                    elif cv and len(cv)==2:
                        import numpy as np
                        bins = np.linspace(df3[cv[0]].min(),df3[cv[0]].max(),11)
                        vals =df3.groupby(pd.cut(df[cv[0]],bins ))[cv[1]].mean()    
                        fig.add_trace(go.Bar(x=bins, y=vals,name='{}-{}'.format(row_s,col_s)),row=r1+1, col=c1+1)
                elif fv== "Heatmap":
                    
                    fig.add_trace(go.Heatmap(x=df3[cv[0]], y=df3[cv[1]], z=df3["val"]),row=r1+1, col=c1+1)
                    
       
    fig2={}
    if fv=='BoxPlots':
        fig.update_traces(boxpoints='all', jitter=.3)
        if pv:
            d1 = df_selected.groupby(pv)[cv[0]].describe().unstack().reset_index()
            d1=  d1.pivot(pv,columns='level_0').reset_index()
            d1.columns = d1.columns.get_level_values(1)
    
        else:
            d1 = df_selected[cv[0]].describe()
        
        fig2 = go.Figure(data=[go.Table(header=dict(values=list(d1.columns),
                fill_color='paleturquoise',
                align='left'),
               cells=dict(values=d1.transpose().values.tolist(),
               fill_color='lavender',
               align='left'))
    ])
    return fig,fig2





if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
