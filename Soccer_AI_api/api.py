from flask import Flask, request, jsonify
import joblib
from Helper_Functions import GetStats, FormatPrediction
import traceback
import pandas as pd
import numpy as np
import json

model = joblib.load("model.pkl")
model_columns = joblib.load("model_columns.pkl")

app = Flask(__name__)

#Request format: {'a_team_title':'','h_team_title':''}
@app.route('/', methods=['POST'])
def index():
    try:
        json_ = request.json
        query_df = pd.DataFrame(json_)
        away_team = query_df.iloc[0]['a_team_title']
        home_team = query_df.iloc[0]['h_team_title']
        match_query = GetStats(away_team,home_team)
        match_query = match_query.reindex(columns=model_columns, fill_value=0)
        prediction = FormatPrediction(list(model.predict(match_query))[0],away_team,home_team)
        return jsonify(prediction)
    except:
        return jsonify({'Error':'Make sure request format is correct'})

if __name__ == '__main__':
    app.run(debug=True)
