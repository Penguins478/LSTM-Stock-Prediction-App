import React, { useState } from 'react';
import axios from 'axios';

const App = () => {
  const [stockTicker, setStockTicker] = useState('AAPL');
  const [predictionGraph, setPredictionGraph] = useState(null);
  const [summaryData, setSummaryData] = useState(null);

  const handlePrediction = async () => {
    try {
      const response = await axios.post('http://localhost:8000/api/predict/', { stock_ticker: stockTicker });
      setPredictionGraph(response.data.graph);
      setSummaryData(response.data.summary);
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div>
      <h1>Stock Trend Prediction App</h1>
      <div>
        <label htmlFor="stockTicker">Enter Stock Ticker:</label>
        <input
          type="text"
          id="stockTicker"
          value={stockTicker}
          onChange={(e) => setStockTicker(e.target.value)} // Update the stockTicker state
        />
        <button onClick={handlePrediction}>Generate Prediction</button>
      </div>
      <div><p>Please wait a bit for the servers to process your request after you choose to generate a prediction.</p></div>
      {summaryData && (
        <div>
          <h2>Summary Statistics</h2>
          <table>
            <thead>
              <tr>
                <th> </th> {/* This is an empty header for the index column */}
                {Object.keys(summaryData).map((key) => (
                  <th key={key}>{key}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.keys(summaryData[Object.keys(summaryData)[0]]).map((key) => (
                <tr key={key}>
                  <td>{key}</td>
                  {Object.keys(summaryData).map((header) => (
                    <td key={header}>{summaryData[header][key]}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {predictionGraph && (
        <div>
          <h2>Data Charts</h2>
          <img src={`data:image/png;base64,${predictionGraph}`} alt="Closing Price vs. Time Chart" />
        </div>
      )}
    </div>
  );
};

export default App;
