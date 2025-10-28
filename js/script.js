const canvas = document.getElementById('waveformCanvas');
const ctx = canvas.getContext('2d');
canvas.width = canvas.offsetWidth;
canvas.height = canvas.offsetHeight;
let fakeConfidence = 0;

let analyser;
let dataArray;
let audioCtx;

async function initMicrophone() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const source = audioCtx.createMediaStreamSource(stream);

    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 1024;

    const bufferLength = analyser.fftSize;
    dataArray = new Uint8Array(bufferLength);

    source.connect(analyser);
    draw();
  } catch (err) {
    console.error('Microphone access denied or not available:', err);
    alert('Please allow microphone access to see the waveform.');
  }
}

function draw() {
  requestAnimationFrame(draw);

  analyser.getByteTimeDomainData(dataArray);

  ctx.fillStyle = '#1e1e1e';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.lineWidth = 2;
  ctx.strokeStyle = '#2ecc71';
  ctx.beginPath();

  const sliceWidth = canvas.width / dataArray.length;
  let x = 0;

  for (let i = 0; i < dataArray.length; i++) {
    const v = dataArray[i] / 128.0; // Normalize from 0–255 to around 1
    const y = v * canvas.height / 2;

    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);

    x += sliceWidth;
  }

  ctx.lineTo(canvas.width, canvas.height / 2);
  ctx.stroke();
}

initMicrophone();

function updateConfidence(confidence) {
  const fill = document.getElementById('confidenceFill');
  const text = document.getElementById('confidenceText');
  fill.style.width = `${confidence}%`;
  text.textContent = `${confidence.toFixed(1)}%`;
}

// Simulate fake confidence values
setInterval(() => {
  fakeConfidence = Math.random() * 100;
  updateConfidence(fakeConfidence);
}, 500);

function updateStatus(rate) {
    const statusText = document.getElementById('statusText');
    const fill = document.getElementById('confidenceFill');
    if (rate >= 80) {
      statusText.textContent = "ALERT";
      statusText.className = "status alert";
      fill.style.backgroundColor = '#e74c3c'; // Red for ALERT
    } else {
      statusText.textContent = "NO ALERT";
      statusText.className = "status safe";
      fill.style.backgroundColor = '#2ecc71'; // Green for NO ALERT
    }
  }
  
  // Update after every 10 millisecond
  setInterval(() => {
    updateStatus(fakeConfidence);
  }, 10);