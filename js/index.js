function shuffle(array) {
    let currentIndex = array.length;
    while (currentIndex !== 0) {
        let randomIndex = Math.floor(Math.random() * currentIndex);
        currentIndex--;
        [array[currentIndex], array[randomIndex]] = [array[randomIndex], array[currentIndex]];
    }
    return array;
}

class Maze {
    constructor(rows = 10, cols = 10) {
        this.rows = rows;
        this.cols = cols;
        // initialize occupancy matrix
        // 0 for empty, MazeCell if occupied
        this.cellMatrix = Array.from({ length: this.rows }, 
            () => Array(this.cols).fill(0));
        this.initDisplay();
    }

    initDisplay(gridSize = 100) {
        let canvasContainer = document.getElementById('canvas-container');
        let oldCanvas = document.getElementById('maze-canvas');
        let newCanvas = document.createElement('canvas');
        canvasContainer.replaceChild(newCanvas, oldCanvas);
        this.canvas = newCanvas;
        this.canvas.setAttribute('id', 'maze-canvas');

        this.canvas.width = this.cols * gridSize;
        this.canvas.height = this.rows * gridSize;
        this.gridSize = gridSize;

        /* Make the heading the same width as the canvas*/
        let heading = document.getElementById('main-heading');
        heading.style.width = window.getComputedStyle(this.canvas).width;

        this.ctx = this.canvas.getContext('2d');
    }

    makeDisplayGrid(gridSize = 100) {

        this.ctx.beginPath();
        this.ctx.strokeStyle = "#000";  // Grid default line color
        this.ctx.lineWidth = 10;
        for (let x = 0; x <= this.canvas.width; x += gridSize) {
            this.ctx.moveTo(x, 0);
            this.ctx.lineTo(x, this.canvas.height);
        }

        for (let y = 0; y <= this.canvas.height; y += gridSize) {
            this.ctx.moveTo(0, y);
            this.ctx.lineTo(this.canvas.width, y);
        }
        this.ctx.stroke();
        this.ctx.closePath();
    }

    placeCell(mazeCell) {
        this.cellMatrix[mazeCell.y][mazeCell.x] = mazeCell;
        return mazeCell;
    }

    // make one maze path from a starting point
    makeMazePath(start) {
        let currentCell = start;
        let prevCell = start;
        do {
            this.placeCell(currentCell);
            prevCell = currentCell;
            currentCell = currentCell.randomStep();
            if (currentCell) {
                currentCell.markCrossing(prevCell);
            }
        } while (currentCell); // returns false when enclosed
    }

    // get a new unenclosed starting point on the existing path
    getNewStart() {
        //let flattened = this.cellMatrix.flat(); 
        return this.cellMatrix.flat().find((c) => (c !== 0) && !c.enclosed());
    }

    // make a maze, drawing lines between centers of cells
    makeMaze(start = new MazeCell(0, 0, this)) {
        // reinitialize
        this.cellMatrix = Array.from({ length: this.rows }, 
            () => Array(this.cols).fill(0));
        this.initDisplay();
        this.makeDisplayGrid();
        
        for (this.makeMazePath(start) ; start ; start = this.getNewStart()) {
            this.makeMazePath(start);
        }

        this.fillRects();
    }

    // fill in uncrossed cells
    fillRects() {
        this.ctx.fillStyle = "black;"
        for (let i = 0 ; i < this.cols ; i++) {
            for (let j = 0 ; j < this.rows ; j++) {
                if (! this.cellMatrix[j][i]) {
                    this.ctx.fillRect(i * this.gridSize, 
                                      j * this.gridSize, 
                                      this.gridSize,
                                      this.gridSize);
                }
            }
        }
    }

}

class MazeCell {

    constructor(x, y, maze) {
        this.x = x;
        this.y = y;
        this.maze = maze;
    }

    static moves = [{ x: 0, y: 1 }, { x: 1, y: 0 }, { x: 0, y: -1 }, { x: -1, y: 0 }];

    // display coordinates for the center of the cell
    displayCenterLoc() {
        let centerDelta = this.maze.gridSize / 2;
        let result = [this.x * this.maze.gridSize + centerDelta, this.y * this.maze.gridSize + centerDelta];
        return result;
    }

    // change the color od grid segments crossed
    markCrossing(fromCell, gridColor = "#fff", pathColor = "#00d") {
 
        // mark the grid crossing
        this.maze.ctx.beginPath();
        this.maze.ctx.strokeStyle = gridColor;
        this.maze.ctx.lineWidth = 10;

        if (this.x < fromCell.x) {
            this.maze.ctx.moveTo(fromCell.x * this.maze.gridSize, fromCell.y * this.maze.gridSize + 5);
            this.maze.ctx.lineTo(fromCell.x * this.maze.gridSize, (fromCell.y + 1) * this.maze.gridSize - 5);
        }
        else if (this.x > fromCell.x) {
            this.maze.ctx.moveTo(this.x * this.maze.gridSize, this.y * this.maze.gridSize + 5);
            this.maze.ctx.lineTo(this.x * this.maze.gridSize, (this.y + 1) * this.maze.gridSize - 5);
        }
        else if (this.y < fromCell.y) {
            this.maze.ctx.moveTo(fromCell.x * this.maze.gridSize + 5, fromCell.y * this.maze.gridSize);
            this.maze.ctx.lineTo((fromCell.x + 1) * this.maze.gridSize - 5, fromCell.y * this.maze.gridSize);
        }
        else if (this.y > fromCell.y) {
            this.maze.ctx.moveTo(this.x * this.maze.gridSize + 5, this.y * this.maze.gridSize);
            this.maze.ctx.lineTo((this.x + 1) * this.maze.gridSize - 5, this.y * this.maze.gridSize);
        }
        this.maze.ctx.stroke();
        this.maze.ctx.closePath();

        // mark the path
        this.maze.ctx.beginPath();
        this.maze.ctx.strokeStyle = pathColor;
        this.maze.ctx.lineWidth = 5;
        this.maze.ctx.moveTo(...fromCell.displayCenterLoc());
        this.maze.ctx.lineTo(...this.displayCenterLoc());
        this.maze.ctx.stroke();
        this.maze.ctx.closePath();

    }

    step(move) {
        return new MazeCell(this.x + move.x, this.y + move.y, this.maze);
    }

    enclosed() {
        return ((this.x === 0 || this.maze.cellMatrix[this.y][this.x - 1]) &&
                (this.x === this.maze.cols - 1 || this.maze.cellMatrix[this.y][this.x + 1]) &&
                (this.y === 0 || this.maze.cellMatrix[this.y - 1][this.x]) &&
                (this.y === this.maze.rows - 1 || this.maze.cellMatrix[this.y + 1][this.x])) ;
    }

    // move at random to an adjacent cell
    // return false if no moves are possible
    randomStep() {
        if (this.enclosed()) {
            return false;
        }
        let newCell = new MazeCell(0, 0, this.maze)
        do {
            newCell = this.step(shuffle(MazeCell.moves)[0]);
        // keep going while illegal
        } while (newCell.x < 0 ||
                 newCell.y < 0 || 
                 newCell.x >= this.maze.cols || 
                 newCell.y >= this.maze.rows ||
                 this.maze.cellMatrix[newCell.y][newCell.x] !== 0);
        return newCell;
    }
}


mz = new Maze();
mz.makeMaze();

