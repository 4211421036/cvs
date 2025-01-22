let img = document.getElementById("video");

ws = new WebSocket("ws://localhost:8089");
ws.onmessage = (event) => {
  let payload = JSON.parse(event.data);
  img.setAttribute("src", `data:image/png;base64,${payload.frame}`);
};
