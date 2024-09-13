document.getElementById('uploadForm').addEventListener('submit', async function(event) {
    event.preventDefault();
    
    const formData = new FormData();
    formData.append('image', document.getElementById('imageInput').files[0]);

    const response = await fetch('/plantcare/upload_image/', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();
    document.getElementById('prediction').innerText = result.plant_name;
    document.getElementById('instructions').innerText = `
        Watering Instructions: ${result.watering_instructions}
        Fertilizer Instructions: ${result.fertilizer_instructions}
        Pesticide Instructions: ${result.pesticide_instructions}
    `;
    document.getElementById('result').style.display = 'block';
});
