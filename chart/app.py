from dash import Dash, html, dcc, callback, Output, Input, no_update, State, set_props
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

class App(Dash):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.figure = go.Figure(data=[go.Candlestick(
            x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close']
        )])
        self.figure.update_layout(xaxis_rangeslider_visible=False)
        self.figure.update_yaxes(fixedrange=True)
        self.graph = dcc.Graph(id='graph-content', figure=self.figure)
        self.layout = (
            html.H1(children='Title of Dash App', style={'textAlign':'center'}),
            # dcc.Dropdown(df.country.unique(), 'Canada', id='dropdown-selection'),
            self.graph
        )


app = App(__name__)

@callback(
    Output('graph-content', 'figure'),
    Input('graph-content', 'relayoutData'),
    State('graph-content', 'figure'),
    prevent_initial_call=True
)
def update_graph(value, figure):
    f = None
    if not value:
        raise PreventUpdate
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
        f = go.Figure(figure)
        f.update_yaxes({'range': [_min * .999, _max * 1.001], 'autorange': False})
    return f or figure
server = app.server
if __name__ == '__main__':
    app.run(debug=True)

