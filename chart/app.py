from dash import Dash, html, dcc, callback, Output, Input, no_update
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import sqlalchemy
from sqlalchemy.sql import text
import logging
import sys
log = logging.getLogger(__name__)
log.addHandler(logging.StreamHandler(sys.stdout))
log.setLevel(logging.INFO)

engine = sqlalchemy.create_engine("sqlite:////app/financial_data.db", execution_options={"sqlite_raw_colnames": True})
df = pd.read_sql_table('stock_data', con=engine)

app = Dash(__name__)
server = app.server
# Requires Dash 2.17.0 or later
fig = go.Figure(data=[go.Candlestick(
    x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']
)])
fig.update_layout(xaxis_rangeslider_visible=False)
fig.update_yaxes(fixedrange=True)
graph = dcc.Graph(id='graph-content', figure=fig)
dragmode = 'pan'
x_range = []
y_range = []
app.layout = (
    html.H1(children='Title of Dash App', style={'textAlign':'center'}),
    # dcc.Dropdown(df.country.unique(), 'Canada', id='dropdown-selection'),
    graph
)
@callback(
    Output('graph-content', 'figure'),
    Input('graph-content', 'relayoutData'),
    prevent_initial_call=True
)
def update_graph(value):
    global fig
    global dragmode
    global x_range
    # dff = df[df.country==value]
    log.info(value)
    if not value:
        log.info('NONE')
        return fig
    if 'dragmode' in value:
        dragmode = value['dragmode']
    if 'xaxis.range[0]' in value:
        with engine.connect() as conn:
            q = text(
                "SELECT max(High) FROM stock_data "
                "WHERE Date BETWEEN :r0 AND :r1"
            )
            _max = conn.execute(q, {'r0': value['xaxis.range[0]'], 'r1': value['xaxis.range[1]']}).scalar_one()
            log.info(f'MAX VAL: {_max}')
            q = text(
                "SELECT min(Low) FROM stock_data "
                "WHERE Date BETWEEN :r0 AND :r1"
            )
            _min = conn.execute(q, {'r0': value['xaxis.range[0]'], 'r1': value['xaxis.range[1]']}).scalar_one()
            log.info(f'MIN VAL: {_min}')
        # x_range = (value['xaxis.range[0]'], value['xaxis.range[1]'])
        # fig.update_xaxes(range=x_range)
        fig.update_yaxes(range=[_min*0.95, _max*1.05])
        # fig.update({'layout': {'dragmode': dragmode}})
        log.info(id(fig))
        return fig

    # if x_range:
    #     fig.update_xaxes(range=x_range)
    # fig.update({'layout': {'dragmode': dragmode}})
    return fig

if __name__ == '__main__':
    app.run(debug=True)
