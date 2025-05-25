/**
 * @fileoverview Stroke Risk Predictor client-side application.
 * Handles form submission, BMI calculation, API communication, and risk visualization.
 */

'use strict';

/**
 * Main application namespace.
 * @namespace
 */
const StrokeRiskApp = {
  /**
   * Chart.js instance for risk factors visualization.
   * @type {?Chart}
   * @private
   */
  riskFactorsChart_: null,

  /**
   * Color palette for charts following the design system.
   * @const {!Array<string>}
   * @private
   */
  CHART_COLORS_: [
    '#CC7B5C', '#D4A27F', '#EBDBBC', '#9C8AA5', '#91A694',
    '#8B9BAE', '#666663', '#BFBFBA', '#E5E4DF', '#F0F0EB'
  ],

  /**
   * Mapping of API field names to human-readable labels.
   * @const {!Object<string, string>}
   * @private
   */
  FEATURE_LABELS_: {
    'age': 'Age',
    'hypertension': 'Hypertension',
    'heart_disease': 'Heart Disease',
    'ever_married': 'Ever Married',
    'residence_type': 'Residence Type',
    'bmi': 'BMI',
    'gender': 'Gender',
    'smoking_status': 'Smoking Status',
    'avg_glucose_level': 'Avg Glucose Level',
    'age_glucose': 'Age × Avg Glucose Level',
    'age_hypertension': 'Age × Hypertension',
    'age_heart_disease': 'Age × Heart Disease',
    'age_squared': 'Age²',
    'glucose_squared': 'Avg Glucose Level²',
    'bmi_age': 'BMI × Age',
    'bmi_glucose': 'BMI × Avg Glucose Level'
  },

  /**
   * Initializes the application when DOM is ready.
   * Sets up event listeners and creates initial empty chart.
   */
  init() {
    // Cache DOM elements
    this.elements_ = {
      form: document.getElementById('risk-form'),
      resultDiv: document.getElementById('prediction-result'),
      resultText: document.getElementById('result-text'),
      calculateBMIButton: document.getElementById('calculate-bmi'),
      bmiCalculator: document.getElementById('bmi-calculator'),
      submitBMIButton: document.getElementById('submit-bmi'),
      bmiInput: document.getElementById('bmi'),
      heightInput: document.getElementById('height'),
      weightInput: document.getElementById('weight')
    };

    // Validate required elements exist
    if (!this.validateElements_()) {
      console.error('Required DOM elements not found');
      return;
    }

    // Set up event listeners
    this.setupEventListeners_();

    // Create initial empty chart
    this.createRiskFactorsChart_({});
  },

  /**
   * Validates that all required DOM elements exist.
   * @return {boolean} True if all elements exist, false otherwise.
   * @private
   */
  validateElements_() {
    return Object.values(this.elements_).every(element => element !== null);
  },

  /**
   * Sets up all event listeners for the application.
   * @private
   */
  setupEventListeners_() {
    this.elements_.calculateBMIButton.addEventListener('click', 
        () => this.toggleBMICalculator_());
    
    this.elements_.submitBMIButton.addEventListener('click', 
        () => this.calculateBMI_());
    
    this.elements_.form.addEventListener('submit', 
        (event) => this.handleFormSubmit_(event));
  },

  /**
   * Toggles the visibility of the BMI calculator.
   * @private
   */
  toggleBMICalculator_() {
    this.elements_.bmiCalculator.classList.toggle('hidden');
  },

  /**
   * Calculates BMI from height and weight inputs.
   * Formula: BMI = weight (kg) / height² (m²)
   * @private
   */
  calculateBMI_() {
    const heightCm = parseFloat(this.elements_.heightInput.value);
    const weight = parseFloat(this.elements_.weightInput.value);
    
    if (!this.isValidNumber_(heightCm) || !this.isValidNumber_(weight)) {
      alert('Please enter valid height and weight values');
      return;
    }
    
    const heightM = heightCm / 100; // Convert cm to meters
    const bmi = weight / (heightM * heightM);
    
    this.elements_.bmiInput.value = bmi.toFixed(1);
    this.elements_.bmiCalculator.classList.add('hidden');
  },

  /**
   * Validates if a value is a valid positive number.
   * @param {*} value - Value to validate.
   * @return {boolean} True if valid positive number.
   * @private
   */
  isValidNumber_(value) {
    return !isNaN(value) && value > 0;
  },

  /**
   * Handles form submission for stroke risk prediction.
   * @param {!Event} event - Form submit event.
   * @private
   */
  async handleFormSubmit_(event) {
    event.preventDefault();
    
    try {
      const formData = this.collectFormData_();
      const processedData = this.preprocessFormData_(formData);
      const result = await this.submitPrediction_(processedData);
      this.displayResults_(result);
    } catch (error) {
      this.handleError_(error);
    }
  },

  /**
   * Collects data from the form.
   * @return {!Object} Form data as key-value pairs.
   * @private
   */
  collectFormData_() {
    const formData = new FormData(this.elements_.form);
    return Object.fromEntries(formData.entries());
  },

  /**
   * Preprocesses form data for API submission.
   * Converts strings to appropriate data types.
   * @param {!Object} data - Raw form data.
   * @return {!Object} Processed data ready for API.
   * @private
   */
  preprocessFormData_(data) {
    const processed = {...data};
    
    // Convert numeric fields
    const numericFields = ['age', 'bmi', 'glucose_level'];
    numericFields.forEach(field => {
      if (processed[field] !== '') {
        processed[field] = parseFloat(processed[field]);
      } else if (field === 'glucose_level') {
        // Remove empty glucose_level field
        delete processed[field];
      }
    });
    
    // Convert integer fields
    const integerFields = ['hypertension', 'heart_disease', 'ever_married', 'residence_type'];
    integerFields.forEach(field => {
      processed[field] = parseInt(processed[field], 10);
    });
    
    return processed;
  },

  /**
   * Submits prediction data to the API.
   * @param {!Object} data - Processed form data.
   * @return {!Promise<!Object>} API response.
   * @throws {Error} If API request fails.
   * @private
   */
  async submitPrediction_(data) {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const result = await response.json();
    
    if (!result.success) {
      throw new Error(result.error || 'Unknown error occurred');
    }
    
    return result;
  },

  /**
   * Displays prediction results to the user.
   * @param {!Object} result - API response containing prediction.
   * @private
   */
  displayResults_(result) {
    const predictionData = result.prediction;
    
    if (!this.isValidPrediction_(predictionData)) {
      throw new Error('Invalid prediction value received');
    }
    
    const riskPercentage = (predictionData.prediction * 100).toFixed(2);
    this.elements_.resultText.textContent = 
        `The estimated stroke risk is ${riskPercentage}%`;
    this.elements_.resultDiv.classList.remove('hidden');
    
    if (predictionData.feature_importances) {
      this.createRiskFactorsChart_(predictionData.feature_importances);
    } else {
      console.warn('Feature importances not provided in the response');
    }
  },

  /**
   * Validates prediction data structure.
   * @param {*} predictionData - Data to validate.
   * @return {boolean} True if valid.
   * @private
   */
  isValidPrediction_(predictionData) {
    return predictionData && 
           typeof predictionData.prediction === 'number' && 
           !isNaN(predictionData.prediction) &&
           predictionData.prediction >= 0 &&
           predictionData.prediction <= 1;
  },

  /**
   * Handles errors during prediction process.
   * @param {!Error} error - Error object.
   * @private
   */
  handleError_(error) {
    console.error('Prediction error:', error);
    this.elements_.resultText.textContent = 
        `An error occurred while processing your request: ${error.message}`;
    this.elements_.resultDiv.classList.remove('hidden');
  },

  /**
   * Creates or updates the risk factors importance chart.
   * @param {!Object} featureImportances - Feature importance scores.
   * @private
   */
  createRiskFactorsChart_(featureImportances) {
    const ctx = document.getElementById('risk-factors-chart').getContext('2d');
    
    // Destroy existing chart if present
    if (this.riskFactorsChart_) {
      this.riskFactorsChart_.destroy();
    }
    
    // Prepare chart data
    const chartData = this.prepareChartData_(featureImportances);
    
    // Create new chart
    this.riskFactorsChart_ = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: chartData.labels,
        datasets: [{
          label: 'Feature Importance',
          data: chartData.values,
          backgroundColor: this.CHART_COLORS_,
          borderColor: this.CHART_COLORS_,
          borderWidth: 1
        }]
      },
      options: this.getChartOptions_()
    });
  },

  /**
   * Prepares data for chart visualization.
   * @param {!Object} featureImportances - Raw feature importance data.
   * @return {{labels: !Array<string>, values: !Array<number>}} Chart data.
   * @private
   */
  prepareChartData_(featureImportances) {
    // Sort by importance and take top 10
    const sortedImportances = Object.entries(featureImportances)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10);
    
    const labels = sortedImportances.map(([feature, _]) => 
        this.FEATURE_LABELS_[feature] || feature);
    
    const values = sortedImportances.map(([_, importance]) => importance);
    
    return {labels, values};
  },

  /**
   * Returns Chart.js configuration options.
   * @return {!Object} Chart options.
   * @private
   */
  getChartOptions_() {
    return {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: 'Importance'
          }
        },
        x: {
          ticks: {
            maxRotation: 45,
            minRotation: -45
          }
        }
      },
      plugins: {
        title: {
          display: true,
          text: 'Top 10 Risk Factors Importance'
        },
        legend: {
          display: false
        },
        tooltip: {
          callbacks: {
            label: (context) => {
              const value = context.parsed.y.toFixed(4);
              return `Importance: ${value}`;
            }
          }
        }
      }
    };
  }
};

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  StrokeRiskApp.init();
});