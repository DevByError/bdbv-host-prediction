/**
 * BDBV Host Prediction - Frontend JavaScript
 */

// Common species suggestions
const COMMON_SPECIES = [
    'Homo_sapiens',
    'Pan_troglodytes',
    'Pan_paniscus',
    'Gorilla_gorilla',
    'Pongo_pygmaeus',
    'Eidolon_helvum',
    'Rousettus_aegyptiacus',
    'Myotis_myotis',
    'Sus_scrofa',
    'Chlorocebus_aethiops',
    'Cercopithecus_ascanius',
    'Macacca_fascicularis',
    'Procavia_capensis',
];

// DOM Elements
const speciesInput = document.getElementById('species-input');
const modelSelect = document.getElementById('model-select');
const virusSelect = document.getElementById('virus-select');
const predictBtn = document.getElementById('predict-btn');
const batchPredictBtn = document.getElementById('batch-predict-btn');
const speciesListInput = document.getElementById('species-list-input');
const loading = document.getElementById('loading');
const resultsCard = document.getElementById('results-card');
const resultsContent = document.getElementById('results-content');
const errorMessage = document.getElementById('error-message');
const batchResults = document.getElementById('batch-results');
const batchTableBody = document.getElementById('batch-table-body');
const suggestionsDiv = document.getElementById('suggestions');
const exampleBtns = document.querySelectorAll('.example-btn');
const apiStatus = document.getElementById('api-status');

// Event Listeners
predictBtn.addEventListener('click', handleSinglePrediction);
batchPredictBtn.addEventListener('click', handleBatchPrediction);
speciesInput.addEventListener('input', handleSpeciesSuggestions);
speciesInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        handleSinglePrediction();
    }
});

exampleBtns.forEach(btn => {
    btn.addEventListener('click', (e) => {
        const species = e.target.dataset.species;
        speciesInput.value = species;
        speciesInput.focus();
        suggestionsDiv.classList.remove('active');
        handleSinglePrediction();
    });
});

// Initialize
checkApiStatus();
setInterval(checkApiStatus, 30000); // Check every 30 seconds

/**
 * Check API Status
 */
async function checkApiStatus() {
    try {
        const response = await fetch('/api/status');
        if (response.ok) {
            apiStatus.textContent = 'Online';
            apiStatus.className = 'status-badge status-online';
        }
    } catch (error) {
        apiStatus.textContent = 'Offline';
        apiStatus.className = 'status-badge status-offline';
    }
}

/**
 * Handle Species Suggestions
 */
function handleSpeciesSuggestions(e) {
    const value = e.target.value.toLowerCase().trim();
    suggestionsDiv.innerHTML = '';

    if (!value) {
        suggestionsDiv.classList.remove('active');
        return;
    }

    const matches = COMMON_SPECIES.filter(species =>
        species.toLowerCase().includes(value)
    );

    if (matches.length === 0) {
        suggestionsDiv.classList.remove('active');
        return;
    }

    matches.slice(0, 5).forEach(species => {
        const item = document.createElement('div');
        item.className = 'suggestion-item';
        item.textContent = species;
        item.addEventListener('click', () => {
            speciesInput.value = species;
            suggestionsDiv.classList.remove('active');
        });
        suggestionsDiv.appendChild(item);
    });

    suggestionsDiv.classList.add('active');
}

/**
 * Handle Single Prediction
 */
async function handleSinglePrediction() {
    const species = speciesInput.value.trim();

    if (!species) {
        showError('Please enter a species name');
        return;
    }

    const model = modelSelect.value;
    const virus = virusSelect.value;

    showLoading(true);
    errorMessage.style.display = 'none';
    resultsCard.style.display = 'none';

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                species: species,
                model: model,
                virus: virus
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Prediction failed');
        }

        displaySingleResult(data);
        showLoading(false);

    } catch (error) {
        showError(error.message);
        showLoading(false);
    }
}

/**
 * Display Single Result
 */
function displaySingleResult(data) {
    const riskClass = `risk-${data.risk_category.toLowerCase()}`;
    const predictionClass = `risk-${data.prediction.toLowerCase()}`;

    const html = `
        <div class="result-item">
            <div>
                <h4>${data.species}</h4>
                <div class="probability-bar">
                    <div class="probability-fill" style="width: ${data.probability * 100}%">
                        ${data.probability_percent}
                    </div>
                </div>
            </div>
            <div class="result-details">
                <div class="result-row">
                    <span class="result-label">Prediction:</span>
                    <span class="risk-badge ${predictionClass}">
                        ${data.prediction}
                    </span>
                </div>
                <div class="result-row">
                    <span class="result-label">Probability:</span>
                    <span class="result-value">${data.probability_percent}</span>
                </div>
                <div class="result-row">
                    <span class="result-label">Risk Level:</span>
                    <span class="risk-badge ${riskClass}">
                        ${data.risk_category}
                    </span>
                </div>
                <div class="result-row">
                    <span class="result-label">Model:</span>
                    <span class="result-value">${data.model}</span>
                </div>
                <div class="result-row">
                    <span class="result-label">Virus:</span>
                    <span class="result-value">${data.virus}</span>
                </div>
            </div>
        </div>
    `;

    resultsContent.innerHTML = html;
    resultsCard.style.display = 'block';
}

/**
 * Handle Batch Prediction
 */
async function handleBatchPrediction() {
    const speciesText = speciesListInput.value.trim();

    if (!speciesText) {
        showError('Please enter at least one species');
        return;
    }

    // Parse species list (comma or newline separated)
    const speciesList = speciesText
        .split(/[\n,]/)
        .map(s => s.trim())
        .filter(s => s.length > 0);

    if (speciesList.length === 0) {
        showError('Please enter valid species names');
        return;
    }

    const model = modelSelect.value;
    const virus = virusSelect.value;

    showLoading(true);
    errorMessage.style.display = 'none';
    batchResults.style.display = 'none';

    try {
        const response = await fetch('/api/batch-predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                species_list: speciesList,
                model: model,
                virus: virus
            })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Batch prediction failed');
        }

        displayBatchResults(data);
        showLoading(false);

    } catch (error) {
        showError(error.message);
        showLoading(false);
    }
}

/**
 * Display Batch Results
 */
function displayBatchResults(data) {
    batchTableBody.innerHTML = '';

    data.results.forEach(result => {
        const riskClass = `risk-${result.risk_category.toLowerCase()}`;
        const predictionClass = `risk-${result.prediction.toLowerCase()}`;

        const row = document.createElement('tr');
        row.innerHTML = `
            <td><strong>${result.species}</strong></td>
            <td>
                <span class="risk-badge ${predictionClass}">
                    ${result.prediction}
                </span>
            </td>
            <td>
                <strong>${result.probability_percent}</strong>
            </td>
            <td>
                <span class="risk-badge ${riskClass}">
                    ${result.risk_category}
                </span>
            </td>
        `;
        batchTableBody.appendChild(row);
    });

    batchResults.style.display = 'block';
}

/**
 * Show Loading Spinner
 */
function showLoading(show) {
    if (show) {
        loading.style.display = 'block';
    } else {
        loading.style.display = 'none';
    }
}

/**
 * Show Error Message
 */
function showError(message) {
    errorMessage.textContent = message;
    errorMessage.style.display = 'block';
    resultsCard.style.display = 'none';
    batchResults.style.display = 'none';
}

/**
 * Format percentage
 */
function formatPercentage(value) {
    return `${(value * 100).toFixed(1)}%`;
}

console.log('BDBV Host Prediction Frontend Loaded ✅');
