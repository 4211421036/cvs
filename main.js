let img = document.getElementById("video");
let canvas = document.getElementById("landmarkCanvas");
let ctx = canvas.getContext("2d");

const ws = new WebSocket("ws://localhost:8089");

ws.onmessage = (event) => {
    let payload = JSON.parse(event.data);

    // Set image frame
    img.setAttribute("src", `data:image/png;base64,${payload.frame}`);

    // Draw landmarks on canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (payload.faces.length > 0) {
        payload.faces.forEach(face => {
            face.forEach(landmark => {
                let x = landmark[0] * canvas.width;
                let y = landmark[1] * canvas.height;
                ctx.beginPath();
                ctx.arc(x, y, 2, 0, 2 * Math.PI);
                ctx.fillStyle = "red";
                ctx.fill();
            });
        });
    }
};

ws.onerror = (error) => {
    console.error("WebSocket error:", error);
};
