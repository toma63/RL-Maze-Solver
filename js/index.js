var canvas = document.getElementById('maze-canvas');
var ctx = canvas.getContext('2d');

var gridWidth = canvas.width / 10;
var gridHeight = canvas.height / 10;

for (var x = 0; x <= canvas.width; x += gridWidth) {
    ctx.moveTo(x, 0);
    ctx.lineTo(x, canvas.height);
}

for (var y = 0; y <= canvas.height; y += gridHeight) {
    ctx.moveTo(0, y);
    ctx.lineTo(canvas.width, y);
}

ctx.strokeStyle = "#ddd";  // Grid line color
ctx.stroke();