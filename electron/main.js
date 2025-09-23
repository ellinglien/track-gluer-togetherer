const { app, BrowserWindow, shell } = require('electron')
const { spawn } = require('child_process')
const path = require('path')

let mainWindow
let flaskProcess

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1000,
    height: 700,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true
    },
    titleBarStyle: 'default',
    show: false
  })

  // Start Flask server
  startFlaskServer()

  // Wait a bit for server to start, then load the page
  setTimeout(() => {
    mainWindow.loadURL('http://localhost:8765')
    mainWindow.show()
  }, 2000)

  // Open external links in default browser
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url)
    return { action: 'deny' }
  })
}

function startFlaskServer() {
  const pythonPath = path.join(__dirname, '..', 'album_merger_env', 'bin', 'python3')
  const scriptPath = path.join(__dirname, '..', 'trackgluer.py')

  flaskProcess = spawn(pythonPath, [scriptPath, '--electron'], {
    cwd: path.join(__dirname, '..')
  })

  flaskProcess.stdout.on('data', (data) => {
    console.log(`Flask: ${data}`)
  })

  flaskProcess.stderr.on('data', (data) => {
    console.log(`Flask Error: ${data}`)
  })
}

function stopFlaskServer() {
  if (flaskProcess) {
    flaskProcess.kill()
    flaskProcess = null
  }
}

app.whenReady().then(createWindow)

app.on('window-all-closed', () => {
  stopFlaskServer()
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow()
  }
})

app.on('before-quit', () => {
  stopFlaskServer()
})