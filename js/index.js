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
        this.initCellMatrix();
    }

    // initialize occupancy matrix
    // 0 for empty, MazeCell if occupied
    initCellMatrix() {
        this.cellMatrix = Array.from({ length: this.rows }, () => Array.from({ length: this.cols }).fill(0));
        //console.log(`rows ${this.rows} cols ${this.cols} 22: ${this.cellMatrix[2][2]}`);
    }

    initDisplay(gridSize = 100, gridColor = "#000", 
                                pathColor = "#00d", 
                                solveColor = "#f00", 
                                goalColor = "#0f0",
                                bgColor = "#fff") {
        this.gridColor = gridColor;
        this.pathColor = pathColor;
        this.solveColor = solveColor;
        this.goalColor = goalColor;
        this.bgColor = bgColor;
        let canvasContainer = document.getElementById('canvas-container');
        let oldCanvas = document.getElementById('maze-canvas');
        let newCanvas = document.createElement('canvas');

        if (oldCanvas !== null) {
            canvasContainer.replaceChild(newCanvas, oldCanvas);
        }
        else {
            canvasContainer.appendChild(newCanvas); 
        }
        
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
        this.ctx.strokeStyle = this.gridColor;  // Grid default line color
        this.ctx.lineWidth = 10;
        for (let x = 0.0 ; x <= this.canvas.width ; x += gridSize) {
            //console.log("gridSize: ", gridSize, "x: ", x, "width: ", this.canvas.width);
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
        let candidates = this.cellMatrix.flat().filter((c) => (c !== 0) && !c.enclosed());
        return shuffle(candidates)[0];
    }

    // make a maze, drawing lines between centers of cells
    makeMaze(gridSize=100, start = new MazeCell(0, 0, this)) {
        // reinitialize
        this.initCellMatrix();
        this.initDisplay(gridSize);
        this.makeDisplayGrid(gridSize);
        
        for (this.makeMazePath(start) ; start ; start = this.getNewStart()) {
            this.makeMazePath(start);
        }

        // for now goal is always the LR corner
        let goalCell = this.cellMatrix[this.rows - 1][this.cols - 1];
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
    RLTrain(passes = 10) {
        for (let pass = 0 ; pass < passes ; pass++) {
            //console.log("training pass: ", pass);
            let cell = this.getRandomCell();
            //console.log("starting from : ", cell);
            let steps = 0;
            do {
                cell = cell.updateState();
                steps++;
            } while (!cell.goal);
            //console.log("goal reached after ", steps, " steps");
            //console.log("cell: ", cell, "epsilon: ", cell.hp.epsilon);
        }
    }

    // solve the maze from the given x and y
    // use the best q value to select moves
    // mark the path in red.
    // time out after maxSteps
    solveFrom(x = 0, y = 0, maxSteps = 1000, color = this.solveColor) {
        this.clearSolve();
        let steps = 0;
        let cell = this.cellMatrix[y][x];
        this.solvePath.push(cell);
        while (!cell.goal && (steps < maxSteps)) {
            steps++
            let newCell = cell.nextState(cell.bestMove());
            newCell.markPath(cell, color);
            cell = newCell;
            this.solvePath.push(cell);
        }
        console.log(`finished after: ${steps} steps`);
    }

    // clear a solution path from the display
    clearSolve(pathColor = this.pathColor) {
        if (!this.solvePath) {
            this.solvePath = [];
            return;
        }
        let cell = this.solvePath[0];
        for (let i = 1 ; i < this.solvePath.length ; i++) {
            let nextCell = this.solvePath[i];
            nextCell.markPath(cell, pathColor);
            cell = nextCell;
        }
        this.solvePath = [];
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
            //console.log("bestMove: ", bestMove);
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
            //console.log("exploring: ", move);
        }
        else {
            //exploit
            move = this.bestMove();
            //console.log("exploiting: ", move);
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
        //console.log("reward: ", reward);
        
        // update q
        let currentQ = this.q[move.name];
        //console.log("currentQ: ", currentQ);
        let newStateQ = newState.q[newState.bestMove().name];
        //console.log("newStateQ: ", newStateQ);
        let newQ = currentQ + this.hp.alpha * (reward + (this.hp.gamma * newStateQ) - currentQ);
        //console.log("newQ: ", newQ);
        this.q[move.name] = newQ;
        //console.log("newState: ", newState);

        return newState;
    }

    // display coordinates for the center of the cell
    displayCenterLoc() {
        let centerDelta = this.maze.gridSize / 2;
        let result = [this.x * this.maze.gridSize + centerDelta, this.y * this.maze.gridSize + centerDelta];
        return result;
    }

    // mark the goal cell in green
    markGoal(goalColor = this.maze.goalColor) { 
        this.goal = true;
        this.maze.ctx.fillStyle = goalColor;
        this.maze.ctx.fillRect(this.x * this.maze.gridSize + 5, 
                               this.y * this.maze.gridSize + 5, 
                               this.maze.gridSize - 10,
                               this.maze.gridSize - 10);
    }

    // change the color of grid segments crossed
    // update legal moves
    markCrossing(fromCell, bgColor = this.maze.bgColor, pathColor = this.maze.Pathcolor) {
 
        // mark the grid crossing
        this.maze.ctx.beginPath();
        this.maze.ctx.strokeStyle = bgColor;
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
        this.markPath(fromCell, pathColor);

    }

    markPath(fromCell, pathColor = this.maze.pathColor) {
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
        //console.log("nextState move: ", move);
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

let settingsForm = document.querySelector("[name='settings']");
let trainForm = document.querySelector("[name='train']");
let solveForm = document.querySelector("[name='solve']");
let maze;

// reset form with defaults
function settingsFormDefaults(cols = 30, rows = 30, grid = 30) {
    settingsForm.columns.value = cols; 
    settingsForm.rows.value = rows; 
    settingsForm.gridSize.value = grid; 
}

function trainFormDefault(passes = 2000) {
    trainForm.passes.value = passes;
}

function solveFormDefaults(startx = 0, starty = 0, limit = 1000) {
    solveForm.startx.value = startx; 
    solveForm.starty.value = starty; 
    solveForm.limit.value = limit; 
}

settingsFormDefaults();
trainFormDefault();
solveFormDefaults();

settingsForm.addEventListener('submit', (event) => {
    event.preventDefault();
    let rows = Number(event.target.rows.value);
    let columns = Number(event.target.columns.value);
    let gridSize = Number(event.target.gridSize.value);

    //console.log('Rows:', rows);
    //console.log('Columns:', columns);
    //console.log('Grid Size:', gridSize);

    maze = new Maze(rows, columns);
    maze.makeMaze(gridSize);

    settingsForm.reset();
    settingsFormDefaults(columns, rows, gridSize);
});

trainForm.addEventListener('submit', (event) => {
    event.preventDefault();
    let passes = Number(event.target.passes.value);
    console.log("passes: ", passes);

    if (!maze) {
        console.error("Error: maze not defined");
    }
    else {
        maze.RLTrain(passes); // make async?
        console.log("training done!");
    }

    trainForm.reset();
    trainFormDefault(passes);
});

solveForm.addEventListener('submit', (event) => {
    event.preventDefault();
    let startx = Number(event.target.startx.value);
    let starty = Number(event.target.starty.value);
    let limit = Number(event.target.limit.value);

    console.log('startx:', startx);
    console.log('starty:', starty);
    console.log('limit:', limit);

    if (!maze) {
        console.error("Error: maze not defined");
    }
    else {
        maze.solveFrom(startx, starty, limit);
        console.log("Done solving!");
    }

    solveForm.reset();
    solveFormDefaults(startx, starty, limit);
});

//mz = new Maze(30, 30);
//mz.makeMaze(30);
//mz.RLTrain(10000);
//mz.solveFrom();

