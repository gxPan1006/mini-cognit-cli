const BOARD_WIDTH = 10;
const BOARD_HEIGHT = 20;

const gameBoard = document.getElementById('game-board');
const scoreDisplay = document.getElementById('score');
const nextPieceDisplay = document.getElementById('next-piece');
const startButton = document.getElementById('start-button');

let board = [];
let score = 0;
let level = 1;
let linesCleared = 0;
let currentPiece = null;
let nextPiece = null;
let currentX = 0;
let currentY = 0;
let gameInterval = null;
let isGameOver = false;
let isPaused = false;

// Each tetromino: array of rotations, each rotation is a 2D array
const SHAPES = {
    I: [
        [[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]],
        [[0,0,1,0],[0,0,1,0],[0,0,1,0],[0,0,1,0]],
        [[0,0,0,0],[0,0,0,0],[1,1,1,1],[0,0,0,0]],
        [[0,1,0,0],[0,1,0,0],[0,1,0,0],[0,1,0,0]]
    ],
    J: [
        [[1,0,0],[1,1,1],[0,0,0]],
        [[0,1,1],[0,1,0],[0,1,0]],
        [[0,0,0],[1,1,1],[0,0,1]],
        [[0,1,0],[0,1,0],[1,1,0]]
    ],
    L: [
        [[0,0,1],[1,1,1],[0,0,0]],
        [[0,1,0],[0,1,0],[0,1,1]],
        [[0,0,0],[1,1,1],[1,0,0]],
        [[1,1,0],[0,1,0],[0,1,0]]
    ],
    O: [
        [[1,1],[1,1]]
    ],
    S: [
        [[0,1,1],[1,1,0],[0,0,0]],
        [[0,1,0],[0,1,1],[0,0,1]],
        [[0,0,0],[0,1,1],[1,1,0]],
        [[1,0,0],[1,1,0],[0,1,0]]
    ],
    T: [
        [[0,1,0],[1,1,1],[0,0,0]],
        [[0,1,0],[0,1,1],[0,1,0]],
        [[0,0,0],[1,1,1],[0,1,0]],
        [[0,1,0],[1,1,0],[0,1,0]]
    ],
    Z: [
        [[1,1,0],[0,1,1],[0,0,0]],
        [[0,0,1],[0,1,1],[0,1,0]],
        [[0,0,0],[1,1,0],[0,1,1]],
        [[0,1,0],[1,1,0],[1,0,0]]
    ]
};

const PIECE_NAMES = ['I', 'J', 'L', 'O', 'S', 'T', 'Z'];
const COLORS = {
    I: 'I', J: 'J', L: 'L', O: 'O', S: 'S', T: 'T', Z: 'Z'
};

function createBoard() {
    board = [];
    gameBoard.innerHTML = '';
    for (let r = 0; r < BOARD_HEIGHT; r++) {
        board.push(Array(BOARD_WIDTH).fill(0));
        for (let c = 0; c < BOARD_WIDTH; c++) {
            const cell = document.createElement('div');
            cell.classList.add('cell');
            gameBoard.appendChild(cell);
        }
    }
}

function drawBoard() {
    for (let r = 0; r < BOARD_HEIGHT; r++) {
        for (let c = 0; c < BOARD_WIDTH; c++) {
            const cell = gameBoard.children[r * BOARD_WIDTH + c];
            cell.className = 'cell';
            if (board[r][c]) {
                cell.classList.add(board[r][c]);
            }
        }
    }
}

function drawPiece() {
    if (!currentPiece) return;
    const shape = currentPiece.shape;
    for (let r = 0; r < shape.length; r++) {
        for (let c = 0; c < shape[r].length; c++) {
            if (shape[r][c]) {
                const boardR = currentY + r;
                const boardC = currentX + c;
                if (boardR >= 0 && boardR < BOARD_HEIGHT && boardC >= 0 && boardC < BOARD_WIDTH) {
                    const cell = gameBoard.children[boardR * BOARD_WIDTH + boardC];
                    cell.classList.add(currentPiece.color);
                }
            }
        }
    }
}

function drawNextPiece() {
    nextPieceDisplay.innerHTML = '';
    if (!nextPiece) return;
    const shape = nextPiece.rotations[0];
    const size = shape.length;
    nextPieceDisplay.style.gridTemplateColumns = `repeat(${size}, 15px)`;
    nextPieceDisplay.style.gridTemplateRows = `repeat(${size}, 15px)`;
    nextPieceDisplay.style.width = (size * 15) + 'px';
    nextPieceDisplay.style.height = (size * 15) + 'px';
    for (let r = 0; r < size; r++) {
        for (let c = 0; c < size; c++) {
            const cell = document.createElement('div');
            cell.classList.add('cell');
            if (shape[r][c]) {
                cell.classList.add(nextPiece.color);
            }
            nextPieceDisplay.appendChild(cell);
        }
    }
}

function randomPiece() {
    const name = PIECE_NAMES[Math.floor(Math.random() * PIECE_NAMES.length)];
    const rotations = SHAPES[name];
    return {
        name,
        rotations,
        rotationIndex: 0,
        shape: rotations[0],
        color: COLORS[name]
    };
}

function isValidPosition(shape, offX, offY) {
    for (let r = 0; r < shape.length; r++) {
        for (let c = 0; c < shape[r].length; c++) {
            if (shape[r][c]) {
                const newR = offY + r;
                const newC = offX + c;
                if (newC < 0 || newC >= BOARD_WIDTH || newR >= BOARD_HEIGHT) {
                    return false;
                }
                if (newR >= 0 && board[newR][newC]) {
                    return false;
                }
            }
        }
    }
    return true;
}

function lockPiece() {
    const shape = currentPiece.shape;
    for (let r = 0; r < shape.length; r++) {
        for (let c = 0; c < shape[r].length; c++) {
            if (shape[r][c]) {
                const boardR = currentY + r;
                const boardC = currentX + c;
                if (boardR < 0) {
                    gameOver();
                    return;
                }
                board[boardR][boardC] = currentPiece.color;
            }
        }
    }
    clearLines();
    spawnPiece();
}

function clearLines() {
    let lines = 0;
    for (let r = BOARD_HEIGHT - 1; r >= 0; r--) {
        if (board[r].every(cell => cell !== 0)) {
            board.splice(r, 1);
            board.unshift(Array(BOARD_WIDTH).fill(0));
            lines++;
            r++; // re-check this row
        }
    }
    if (lines > 0) {
        const points = [0, 100, 300, 500, 800];
        score += points[lines] * level;
        linesCleared += lines;
        level = Math.floor(linesCleared / 10) + 1;
        scoreDisplay.textContent = score;
        updateSpeed();
    }
}

function updateSpeed() {
    if (gameInterval) {
        clearInterval(gameInterval);
        const speed = Math.max(100, 800 - (level - 1) * 70);
        gameInterval = setInterval(tick, speed);
    }
}

function spawnPiece() {
    currentPiece = nextPiece || randomPiece();
    nextPiece = randomPiece();
    currentPiece.rotationIndex = 0;
    currentPiece.shape = currentPiece.rotations[0];
    currentX = Math.floor((BOARD_WIDTH - currentPiece.shape[0].length) / 2);
    currentY = -1;

    if (!isValidPosition(currentPiece.shape, currentX, currentY)) {
        // Try one row up
        currentY = -2;
        if (!isValidPosition(currentPiece.shape, currentX, currentY)) {
            gameOver();
            return;
        }
    }

    drawNextPiece();
}

function moveDown() {
    if (isValidPosition(currentPiece.shape, currentX, currentY + 1)) {
        currentY++;
        return true;
    }
    lockPiece();
    return false;
}

function moveLeft() {
    if (isValidPosition(currentPiece.shape, currentX - 1, currentY)) {
        currentX--;
    }
}

function moveRight() {
    if (isValidPosition(currentPiece.shape, currentX + 1, currentY)) {
        currentX++;
    }
}

function rotate() {
    const nextRotation = (currentPiece.rotationIndex + 1) % currentPiece.rotations.length;
    const nextShape = currentPiece.rotations[nextRotation];

    // Wall kick: try offsets 0, -1, +1, -2, +2
    const kicks = [0, -1, 1, -2, 2];
    for (const kick of kicks) {
        if (isValidPosition(nextShape, currentX + kick, currentY)) {
            currentPiece.rotationIndex = nextRotation;
            currentPiece.shape = nextShape;
            currentX += kick;
            return;
        }
    }
}

function hardDrop() {
    while (isValidPosition(currentPiece.shape, currentX, currentY + 1)) {
        currentY++;
        score += 2;
    }
    scoreDisplay.textContent = score;
    lockPiece();
}

function drawGhost() {
    if (!currentPiece) return;
    let ghostY = currentY;
    while (isValidPosition(currentPiece.shape, currentX, ghostY + 1)) {
        ghostY++;
    }
    if (ghostY === currentY) return;
    const shape = currentPiece.shape;
    for (let r = 0; r < shape.length; r++) {
        for (let c = 0; c < shape[r].length; c++) {
            if (shape[r][c]) {
                const boardR = ghostY + r;
                const boardC = currentX + c;
                if (boardR >= 0 && boardR < BOARD_HEIGHT && boardC >= 0 && boardC < BOARD_WIDTH) {
                    const cell = gameBoard.children[boardR * BOARD_WIDTH + boardC];
                    if (!cell.classList.contains(currentPiece.color)) {
                        cell.classList.add('ghost');
                    }
                }
            }
        }
    }
}

function tick() {
    if (isGameOver || isPaused) return;
    moveDown();
    render();
}

function render() {
    drawBoard();
    drawGhost();
    drawPiece();
}

function gameOver() {
    isGameOver = true;
    if (gameInterval) {
        clearInterval(gameInterval);
        gameInterval = null;
    }
    startButton.textContent = 'Restart';
    startButton.disabled = false;

    // Draw game over overlay
    const overlay = document.createElement('div');
    overlay.id = 'game-over-overlay';
    overlay.innerHTML = `<div class="game-over-text">GAME OVER</div><div class="game-over-score">Score: ${score}</div>`;
    gameBoard.style.position = 'relative';
    gameBoard.appendChild(overlay);
}

function startGame() {
    // Remove overlay if exists
    const overlay = document.getElementById('game-over-overlay');
    if (overlay) overlay.remove();

    isGameOver = false;
    isPaused = false;
    score = 0;
    level = 1;
    linesCleared = 0;
    scoreDisplay.textContent = '0';

    if (gameInterval) clearInterval(gameInterval);

    createBoard();
    nextPiece = randomPiece();
    spawnPiece();
    render();

    const speed = Math.max(100, 800 - (level - 1) * 70);
    gameInterval = setInterval(tick, speed);
    startButton.textContent = 'Restart';
}

// Keyboard controls
document.addEventListener('keydown', (e) => {
    if (isGameOver || isPaused) {
        if (e.key === 'p' || e.key === 'P') {
            isPaused = false;
        }
        return;
    }

    switch (e.key) {
        case 'ArrowLeft':
        case 'a':
            moveLeft();
            render();
            e.preventDefault();
            break;
        case 'ArrowRight':
        case 'd':
            moveRight();
            render();
            e.preventDefault();
            break;
        case 'ArrowDown':
        case 's':
            moveDown();
            score += 1;
            scoreDisplay.textContent = score;
            render();
            e.preventDefault();
            break;
        case 'ArrowUp':
        case 'w':
            rotate();
            render();
            e.preventDefault();
            break;
        case ' ':
            hardDrop();
            render();
            e.preventDefault();
            break;
        case 'p':
        case 'P':
            isPaused = !isPaused;
            break;
    }
});

startButton.addEventListener('click', startGame);

// Initialize empty board
createBoard();
