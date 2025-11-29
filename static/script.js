const canvas = document.getElementById("drawCanvas");
const ctx = canvas.getContext("2d");
const predictionDiv = document.getElementById("prediction");
const clearBtn = document.getElementById("clearBtn");

// Canvas settings
canvas.width = 128;
canvas.height = 128;
ctx.fillStyle = "black";     // black background
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.strokeStyle = "white";   // white brush
ctx.lineWidth = 15;           // thick brush
ctx.lineCap = "round";

let drawing = false;

// Mouse events
canvas.addEventListener("mousedown", (e) => {
    drawing = true;
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
});

canvas.addEventListener("mousemove", (e) => {
    if (!drawing) return;
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(e.offsetX, e.offsetY);
});

canvas.addEventListener("mouseup", () => drawing = false);
canvas.addEventListener("mouseout", () => drawing = false);

// Clear canvas
clearBtn.addEventListener("click", () => {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    predictionDiv.textContent = "Prediction: ";
});

// Predict while drawing (continuous)
let timeoutId;
canvas.addEventListener("mousemove", () => {
    if (timeoutId) clearTimeout(timeoutId);
    timeoutId = setTimeout(sendPrediction, 200); // send prediction every 200ms
});

async function sendPrediction() {
    const dataURL = canvas.toDataURL("image/png");
    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ image: dataURL })
        });
        const data = await response.json();
        if (data.prediction !== undefined) {
            predictionDiv.textContent = `Prediction: ${data.prediction}`;
        } else {
            predictionDiv.textContent = `Error: ${data.error}`;
        }
    } catch (err) {
        predictionDiv.textContent = `Error: ${err}`;
    }
}
