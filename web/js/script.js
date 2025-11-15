const canvas = document.getElementById('waveformCanvas');
const ctx = canvas.getContext('2d');
canvas.width = canvas.offsetWidth;
canvas.height = canvas.offsetHeight;

async function drawWave() {
  try {
    const r = await fetch('/waveform.json', {cache: 'no-store'});
    if (!r.ok) return;
    const j = await r.json();
    const pts = j.points;
    const ctx = canvas.getContext('2d');
    const w = canvas.width, h = canvas.height;
    ctx.clearRect(0,0,w,h);
    ctx.beginPath();
    for (let i=0;i<pts.length;i++){
      const x = Math.floor(i * w / pts.length);
      const y = (1 - (pts[i] * 0.5 + 0.5)) * h; // map -1..1 -> 0..h
      if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    }
    ctx.strokeStyle = '#0b6'; ctx.lineWidth = 1;
    ctx.stroke();
  } catch(e) { console.error(e); }
}

// Poll every 1s (or match your recording cadence)
setInterval(drawWave, 1000);
drawWave();

function updateConfidence(confidence) {
  const fill = document.getElementById('confidenceFill');
  const text = document.getElementById('confidenceText');
  fill.style.width = `${confidence}%`;
  text.textContent = `${confidence.toFixed(1)}%`;
} 

// ectract data from json file
async function fetchStatus() {
  try {
    const response = await fetch('/results.json');
    const data = await response.json();
    console.log(data);
    
    updateConfidence(data.confidence_percent);
    updateStatus(data.predicted_label);
  } catch (err) {
    console.error('Error fetching json file:', err);
  }
}

setInterval(() => {
  fetchStatus()
}, 1000);

function updateStatus(label) {
    const statusText = document.getElementById('statusText');
    const fill = document.getElementById('confidenceFill');
    if (label == 1) {
      statusText.textContent = "ALERT";
      statusText.className = "status alert";
      fill.style.backgroundColor = '#ff0000'; // Red for ALERT
    } else {
      statusText.textContent = "NO ALERT";
      statusText.className = "status safe";
      fill.style.backgroundColor = '#2ecc71'; // Green for NO ALERT
    }
  }