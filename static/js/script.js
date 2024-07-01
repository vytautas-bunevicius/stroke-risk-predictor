document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('risk-form');
    const resultDiv = document.getElementById('prediction-result');
    const resultText = document.getElementById('result-text');
    const calculateBMIButton = document.getElementById('calculate-bmi');
    const bmiCalculator = document.getElementById('bmi-calculator');
    const submitBMIButton = document.getElementById('submit-bmi');
    const bmiInput = document.getElementById('bmi');
    let riskFactorsChart;

    // Create the initial empty chart on page load
    createRiskFactorsChart({});

    calculateBMIButton.addEventListener('click', function () {
        bmiCalculator.classList.toggle('hidden');
    });

    submitBMIButton.addEventListener('click', function () {
        const height = parseFloat(document.getElementById('height').value) / 100; // convert cm to m
        const weight = parseFloat(document.getElementById('weight').value);
        if (height && weight) {
            const bmi = weight / (height * height);
            bmiInput.value = bmi.toFixed(1);
            bmiCalculator.classList.add('hidden');
        }
    });

    form.addEventListener('submit', function (e) {
        e.preventDefault();
        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());

        // Convert numeric strings to numbers
        ['age', 'bmi', 'glucose_level'].forEach(key => {
            if (data[key] !== '') {
                data[key] = parseFloat(data[key]);
            } else if (key === 'glucose_level') {
                // If glucose_level is empty, remove it from the data object
                delete data[key];
            }
        });

        // Convert boolean strings to integers
        ['hypertension', 'heart_disease', 'ever_married', 'residence_type'].forEach(key => {
            data[key] = parseInt(data[key]);
        });

        fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        })
            .then(response => {
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                return response.json();
            })
            .then(result => {
                console.log('Received result:', result);
                if (result.success) {
                    const predictionData = result.prediction; // Access the nested prediction object
                    if (typeof predictionData.prediction === 'number' && !isNaN(predictionData.prediction)) {
                        const riskPercentage = (predictionData.prediction * 100).toFixed(2);
                        resultText.textContent = `The estimated stroke risk is ${riskPercentage}%`;
                        resultDiv.classList.remove('hidden');
                        if (predictionData.feature_importances) {
                            createRiskFactorsChart(predictionData.feature_importances);
                        } else {
                            console.warn('Feature importances not provided in the response');
                        }
                    } else {
                        console.error('Invalid prediction value received:', predictionData.prediction);
                        throw new Error('Invalid prediction value received');
                    }
                } else {
                    console.error('Error in response:', result.error);
                    throw new Error(result.error || 'Unknown error occurred');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                resultText.textContent = `An error occurred while processing your request: ${error.message}`;
                resultDiv.classList.remove('hidden');
            });
    });

    function createRiskFactorsChart(featureImportances) {
        const ctx = document.getElementById('risk-factors-chart').getContext('2d');

        if (riskFactorsChart) {
            riskFactorsChart.destroy();
        }

        const sortedImportances = Object.entries(featureImportances)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 10);

        const labels = sortedImportances.map(item => {
            switch (item[0]) {
                case 'age':
                    return 'Age';
                case 'hypertension':
                    return 'Hypertension';
                case 'heart_disease':
                    return 'Heart Disease';
                case 'ever_married':
                    return 'Ever Married';
                case 'residence_type':
                    return 'Residence Type';
                case 'bmi':
                    return 'BMI';
                case 'gender':
                    return 'Gender';
                case 'smoking_status':
                    return 'Smoking Status';
                case 'avg_glucose_level':
                    return 'Avg Glucose Level';
                case 'age_glucose':
                    return 'Age * Avg Glucose Level';
                case 'age_hypertension':
                    return 'Age * Hypertension';
                case 'age_heart_disease':
                    return 'Age * Heart Disease';
                case 'age_squared':
                    return 'Age Squared';
                case 'glucose_squared':
                    return 'Avg Glucose Level Squared';
                case 'bmi_age':
                    return 'BMI * Age';
                case 'bmi_glucose':
                    return 'BMI * Avg Glucose Level';
                default:
                    return item[0]; // For any other feature not explicitly mentioned
            }
        });

        const data = sortedImportances.map(item => item[1]);

        const COLORS = ["#CC7B5C", "#D4A27F", "#EBDBBC", "#9C8AA5", "#91A694", "#8B9BAE", "#666663", "#BFBFBA", "#E5E4DF", "#F0F0EB"];

        riskFactorsChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Feature Importance',
                    data: data,
                    backgroundColor: COLORS,
                    borderColor: COLORS,
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Importance'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: ''
                        },
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
                    }
                }
            }
        });
    }
});