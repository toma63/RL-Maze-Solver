class Maze {
    constructor(rows = 10, cols = 10) {
        this.rows = rows;
        this.cols = cols;
    }

    display(gridSize = 100) {
        this.canvas = document.getElementById('maze-canvas');
        this.canvas.width = this.cols * gridSize;
        this.canvas.height = this.rows * gridSize;
        this.gridSize = gridSize;

        /* Make the heading the same width as the canvas*/
        let heading = document.getElementById('main-heading');
        heading.style.width = window.getComputedStyle(this.canvas).width;

        this.ctx = this.canvas.getContext('2d');

        for (let x = 0; x <= this.canvas.width; x += gridSize) {
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, this.canvas.height);
        }

        for (let y = 0; y <= this.canvas.height; y += gridSize) {
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(this.canvas.width, y);
        }

        this.ctx.strokeStyle = "#ddd";  // Grid default line color
        this.ctx.stroke();
    }
}

mz = new Maze();
mz.display();


