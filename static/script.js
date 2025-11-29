const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);

let drawing = false;

canvas.addEventListener("mousedown", () => drawing = true);
canvas.addEventListener("mouseup", () => drawing = false);
canvas.addEventListener("mousemove", draw);

function draw(e) {
    if (!drawing) return;
    ctx.fillStyle = "white";
    ctx.beginPath();
    ctx.arc(e.offsetX, e.offsetY, 10, 0, Math.PI*2);
    ctx.fill();
}

document.getElementById("clear").addEventListener("click", () => {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
});

document.getElementById("predict").addEventListener("click", () => {
    const data = canvas.toDataURL("image/png");
    fetch("/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({image: data})
    })
    .then(res => res.json())
    .then(data => {
        if(data.prediction !== undefined){
            document.getElementById("result").innerText = "Prediction: " + data.prediction;
        } else {
            document.getElementById("result").innerText = "Error: " + data.error;
        }
    });
});