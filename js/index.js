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
    makeMaze(gridSize=100, start = new MazeCell(0, 0, this)) {
        // reinitialize
        this.cellMatrix = Array.from({ length: this.rows }, 
            () => Array(this.cols).fill(0));
        this.initDisplay(gridSize);
        this.makeDisplayGrid(gridSize);
        
        for (this.makeMazePath(start) ; start ; start = this.getNewStart()) {
            this.makeMazePath(start);
        }

        // for now goal is always the LR corner
        let goalCell = this.cellMatrix[this.cols - 1][this.rows - 1];
        goalCell.markGoal();

        this.fillRects();  // currently don't expect unfilled cells
    }

    // get a random cell
    getRandomCell() {
        let rY = Math.floor(this.rows * Math.random());
        let rX = Math.floor(this.cols * Math.random());
        return this.cellMatrix[rY][rX];
    }

    // run training by starting multiple passes at random places
    // for each pass:
    //   update q values until the goal is reached
    RLTrain(passes = 10, start = this.getRandomCell()) {
        for (let pass = 0 ; pass < passes ; pass++) {
            console.log("pass: ", pass);
            let cell = start;
            do {
                cell = cell.updateState();
            } while (!cell.goal);
        }
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

class RLHyperP {
    constructor(epsilon = 0.3, 
                epsilon_decay = 0.99, 
                alpha = 0.5, 
                gamma = 0.9, 
                rIllegal = -0.75, 
                rLegal = -0.1, 
                rGoal = 10) {
        this.epsilon = epsilon;
        this.epsilon_decay = epsilon_decay;
        this.alpha = alpha;
        this.gamma = gamma;
        this.rIllegal = rIllegal;
        this.rLegal = rLegal;
        this.rGoal = rGoal;
    }
}

class MazeCell {

    constructor(x, y, maze, hp = new RLHyperP()) {
        this.x = x;
        this.y = y;
        this.maze = maze;
        this.legal = {n: false, s: false, e: false, w: false};
        this.q = {n: 0, s:0, e:0, w:0};
        this.goal = false; // goal cell
        this.hp = hp;
    }

    static moves = [{ name: 's', x: 0, y: 1 }, 
                    { name: 'e', x: 1, y: 0 }, 
                    { name: 'n', x: 0, y: -1 }, 
                    { name: 'w', x: -1, y: 0 }];

    // select the best  move based on the highest q value
    bestMove() {
        let qArr = Object.entries(this.q); // array of pairs [name, q]
        // if all are zero, random
        if (qArr.every(val => val[1] === 0)) {
            return this.getRandomMove();
        }
        else {

            // sort on q values in descending order and pick first
            let moveName = qArr.sort((a, b) => b[1] - a[1])[0][0];
            let bestMove = MazeCell.moves.find(x => x.name === moveName);
            console.log("bestMove: ", bestMove);
            return bestMove;
        }
    }

    // update the state based on the Bellman equation
    // return the new cell
    updateState() {
        // select move based on epsilon greedy
        let move = null; // a move object as in the static moves 
        let reward = 0;
        let newState = this; // no move by default, stay if illegal move generated
        if (Math.random() < this.hp.epsilon) {
            // explore
            move = this.getRandomMove();
            console.log("exploring: ", move);
        }
        else {
            //exploit
            move = this.bestMove();
            console.log("exploiting: ", move);
        }
        // update epsilon
        this.hp.epsilon *= this.hp.epsilon_decay;

        // compute reward
        if (this.legal[move.name]) {
            newState = this.nextState(move);
            if (newState.goal) {
                reward = this.hp.rGoal;
            } 
            else {
                reward = this.hp.rLegal;
            }
        } 
        else {
                reward = this.hp.rIllegal;
        }
        console.log("reward: ", reward);
        
        // update q
        let currentQ = this.q[move.name];
        console.log("currentQ: ", currentQ);
        let newStateQ = newState.q[newState.bestMove().name];
        console.log("newStateQ: ", newStateQ);
        let newQ = currentQ + this.hp.alpha * (reward + (this.hp.gamma * newStateQ) - currentQ);
        console.log("newQ: ", newQ);
        this.q[move.name] = newQ;
        console.log("newState: ", newState);

        return newState;
    }

    // display coordinates for the center of the cell
    displayCenterLoc() {
        let centerDelta = this.maze.gridSize / 2;
        let result = [this.x * this.maze.gridSize + centerDelta, this.y * this.maze.gridSize + centerDelta];
        return result;
    }

    // mark the goal cell in green
    markGoal(goalColor = '#0f0') { 
        this.goal = true;
        this.maze.ctx.fillStyle = goalColor;
        this.maze.ctx.fillRect(this.x * this.maze.gridSize + 5, 
                               this.y * this.maze.gridSize + 5, 
                               this.maze.gridSize - 10,
                               this.maze.gridSize - 10);
    }

    // change the color of grid segments crossed
    // update legal moves
    markCrossing(fromCell, gridColor = "#fff", pathColor = "#00d") {
 
        // mark the grid crossing
        this.maze.ctx.beginPath();
        this.maze.ctx.strokeStyle = gridColor;
        this.maze.ctx.lineWidth = 10;

        if (this.x < fromCell.x) {
            this.maze.ctx.moveTo(fromCell.x * this.maze.gridSize, fromCell.y * this.maze.gridSize + 5);
            this.maze.ctx.lineTo(fromCell.x * this.maze.gridSize, (fromCell.y + 1) * this.maze.gridSize - 5);
            fromCell.legal.w = true;
            this.legal.e = true;
        }
        else if (this.x > fromCell.x) {
            this.maze.ctx.moveTo(this.x * this.maze.gridSize, this.y * this.maze.gridSize + 5);
            this.maze.ctx.lineTo(this.x * this.maze.gridSize, (this.y + 1) * this.maze.gridSize - 5);
            fromCell.legal.e = true;
            this.legal.w = true;
        }
        else if (this.y < fromCell.y) {
            this.maze.ctx.moveTo(fromCell.x * this.maze.gridSize + 5, fromCell.y * this.maze.gridSize);
            this.maze.ctx.lineTo((fromCell.x + 1) * this.maze.gridSize - 5, fromCell.y * this.maze.gridSize);
            fromCell.legal.n = true;
            this.legal.s = true;
        }
        else if (this.y > fromCell.y) {
            this.maze.ctx.moveTo(this.x * this.maze.gridSize + 5, this.y * this.maze.gridSize);
            this.maze.ctx.lineTo((this.x + 1) * this.maze.gridSize - 5, this.y * this.maze.gridSize);
            fromCell.legal.s = true;
            this.legal.n = true;
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

    nextState(move) {
        console.log("nextState move: ", move);
        return this.maze.cellMatrix[this.y + move.y][this.x + move.x];
    }

    enclosed() {
        return ((this.x === 0 || this.maze.cellMatrix[this.y][this.x - 1]) &&
                (this.x === this.maze.cols - 1 || this.maze.cellMatrix[this.y][this.x + 1]) &&
                (this.y === 0 || this.maze.cellMatrix[this.y - 1][this.x]) &&
                (this.y === this.maze.rows - 1 || this.maze.cellMatrix[this.y + 1][this.x])) ;
    }

    // select a random move
    getRandomMove() {
        return shuffle(MazeCell.moves)[0];
    }

    // move at random to an adjacent cell
    // return false if no moves are possible
    randomStep() {
        if (this.enclosed()) {
            return false;
        }
        let newCell = new MazeCell(0, 0, this.maze)
        do {
            newCell = this.step(this.getRandomMove());
        // keep going while illegal
        } while (newCell.x < 0 ||
                 newCell.y < 0 || 
                 newCell.x >= this.maze.cols || 
                 newCell.y >= this.maze.rows ||
                 this.maze.cellMatrix[newCell.y][newCell.x] !== 0);
        return newCell;
    }
}


mz = new Maze(30, 30);
mz.makeMaze(30);

