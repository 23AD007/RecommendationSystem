async function getRecommendation() {
    const button = document.querySelector('button');
    const originalText = button.textContent;
    button.textContent = 'Loading...';
    button.disabled = true;

    try {
        const payload = {
            product_category: document.getElementById("productCategory").value,
            fragility_score: parseFloat(document.getElementById("fragility").value),
            sustainability_priority: parseFloat(document.getElementById("sustainability").value),
            durability_requirement: parseFloat(document.getElementById("durability").value),
            material_cost: parseFloat(document.getElementById("materialCost").value),
            max_packaging_cost: parseFloat(document.getElementById("maxCost").value),
            innovation_level: parseFloat(document.getElementById("innovation").value)
        };

        const response = await fetch("http://localhost:5000/api/recommend-materials", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-API-Key": "packaging-api-key-2024"
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (data.status !== "success") {
            throw new Error(data.message || "API request failed");
        }

        displayResults(data);

    } catch (error) {
        alert('Error: ' + (error.message || 'An error occurred while getting recommendations'));
    } finally {
        button.textContent = originalText;
        button.disabled = false;
    }
}

function displayResults(data) {
    const resultsDiv = document.getElementById("results");
    const materialsDiv = document.getElementById("materials");

    materialsDiv.innerHTML = '';

    // Display model info
    const modelInfo = data.model_info || {};
    const modelDiv = document.createElement('div');
    modelDiv.className = 'model-info';
    modelDiv.innerHTML = `<strong>Model Used:</strong> ${modelInfo.model || 'Unknown'} (${modelInfo.confidence_source || ''})`;
    materialsDiv.appendChild(modelDiv);

    if (data.recommendations && data.recommendations.length > 0) {
        data.recommendations.forEach(material => {
            const materialDiv = document.createElement('div');
            materialDiv.className = 'material-item';
            materialDiv.innerHTML = `
                <strong>${material.material}</strong><br>
                Confidence: ${material.confidence}%<br>
                ${material.reason}
            `;
            materialsDiv.appendChild(materialDiv);
        });
    } else {
        materialsDiv.innerHTML += '<p>No recommendations found.</p>';
    }

    resultsDiv.style.display = 'block';
}
