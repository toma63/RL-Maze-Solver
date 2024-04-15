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
    }

    makeDisplayGrid(gridSize = 100) {
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

    placeCell(mazeCell) {
        this.cellMatrix[mazeCell.y][mazeCell.x] = mazeCell;
        return mazeCell;
    }

    makeMaze(start = new MazeCell(0, 0, this)) {
        let currentCell = start;
        do {
            this.placeCell(currentCell);
            currentCell = currentCell.randomStep();
            console.log(currentCell);
        } while (currentCell);
    }
}

class MazeCell {

    constructor(x, y, maze) {
        this.x = x;
        this.y = y;
        this.maze = maze;
    }

    static moves = [{ x: 0, y: 1 }, { x: 1, y: 0 }, { x: 0, y: -1 }, { x: -1, y: 0 }];

    step(move) {
        return new MazeCell(this.x + move.x, this.y + move.y, this.maze);
    }

    enclosed() {
        return ((this.x > 0 && this.maze.cellMatrix[this.y][this.x - 1]) &&
                (this.x < this.maze.cols - 1 && this.maze.cellMatrix[this.y][this.x + 1]) &&
                (this.y > 0 && this.maze.cellMatrix[this.y - 1][this.x]) &&
                (this.y < this.maze.rows - 1 && this.maze.cellMatrix[this.y + 1][this.x])) ;
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
mz.makeDisplayGrid();


