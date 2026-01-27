// Truth Guardian Content Script
console.log("Truth Guardian: Active and protecting...");

// --- Premium CSS Styles ---
const INJECT_CSS = `
    .truth-guardian-marker {
        position: relative;
        display: inline-block;
        border-bottom: 2px dashed rgba(59, 130, 246, 0.4);
        cursor: help;
        transition: background 0.2s;
    }

    .truth-guardian-marker:hover {
        background: rgba(59, 130, 246, 0.1);
    }

    .truth-guardian-icon {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 16px;
        height: 16px;
        margin-left: 4px;
        vertical-align: text-top;
        animation: tg-fade-in 0.4s ease-out;
    }

    @keyframes tg-fade-in {
        from { opacity: 0; transform: scale(0.8); }
        to { opacity: 1; transform: scale(1); }
    }

    .tg-tooltip {
        visibility: hidden;
        opacity: 0;
        position: absolute;
        bottom: calc(100% + 10px);
        left: 50%;
        transform: translateX(-50%) translateY(10px);
        
        /* Glassmorphism */
        background: rgba(30, 41, 59, 0.95);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        
        color: #f8fafc;
        width: 280px;
        padding: 16px;
        border-radius: 16px;
        box-shadow: 
            0 20px 25px -5px rgba(0, 0, 0, 0.3), 
            0 8px 10px -6px rgba(0, 0, 0, 0.3),
            0 0 0 1px rgba(255, 255, 255, 0.1);
        
        z-index: 2147483647; /* Max z-index */
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
        font-size: 13px;
        line-height: 1.5;
        text-align: left;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
        pointer-events: none; /* Prevent blocking interactions unless we want interactive tooltip */
    }

    .truth-guardian-marker:hover .tg-tooltip {
        visibility: visible;
        opacity: 1;
        transform: translateX(-50%) translateY(0);
        pointer-events: auto;
    }

    /* Tooltip Arrow */
    .tg-tooltip::after {
        content: '';
        position: absolute;
        top: 100%;
        left: 50%;
        transform: translateX(-50%);
        border-width: 6px;
        border-style: solid;
        border-color: rgba(30, 41, 59, 0.95) transparent transparent transparent;
    }

    .tg-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
        padding-bottom: 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }

    .tg-badge {
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .tg-real { background: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
    .tg-fake { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
    .tg-neutral { background: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.4); }

    .tg-score {
        font-size: 20px;
        font-weight: 800;
        color: white;
    }
    .tg-score-label {
        font-size: 9px;
        text-transform: uppercase;
        color: #94a3b8;
        display: block;
        margin-bottom: 2px;
    }

    .tg-body {
        color: #cbd5e1;
        margin-bottom: 8px;
    }
    
    .tg-footer {
        font-size: 10px;
        color: #64748b;
        display: flex;
        justify-content: flex-end;
        align-items: center;
        gap: 4px;
    }
`;

// Inject Styles
const styleEl = document.createElement('style');
styleEl.textContent = INJECT_CSS;
document.head.appendChild(styleEl);


// --- Core Logic ---

// 1. Listen for background messages (Context Menu)
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "analyzeSelection") {
    const selection = window.getSelection();
    if (selection.rangeCount > 0) {
      const range = selection.getRangeAt(0);
      const container = document.createElement("span");
      container.className = "tg-highlight-loading";
      container.style.backgroundColor = "rgba(59, 130, 246, 0.2)";
      range.surroundContents(container);

      // Perform Analysis
      analyzeText(request.text, container);
    }
  }
});

// 2. Automatic Page Scanning (Debounced)
let scanTimeout;
// window.addEventListener('scroll', () => {
//     clearTimeout(scanTimeout);
//     scanTimeout = setTimeout(scanPage, 3000); // Less aggressive
// });

// Initial scan with delay
// setTimeout(scanPage, 2000);


async function analyzeText(text, element) {
  try {
    const response = await fetch('http://127.0.0.1:5000/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        content: text,
        url: window.location.href
      })
    });

    const data = await response.json();
    injectTooltip(element, data);
    element.style.backgroundColor = "transparent"; // Remove loading highlight
  } catch (err) {
    console.error("Analysis Failed:", err);
    element.style.backgroundColor = "rgba(239, 68, 68, 0.2)"; // Error state
  }
}

function injectTooltip(element, data) {
  // Add marker class
  element.classList.add('truth-guardian-marker');

  // Determine Color/Icon
  const isFake = data.classification === "Fake News";
  const badgeClass = isFake ? "tg-fake" : "tg-real";
  const iconColor = isFake ? "#ef4444" : "#10b981";

  const tooltipHTML = `
        <div class="tg-tooltip">
            <div class="tg-header">
                <div>
                    <span class="tg-score-label">Credibility Score</span>
                    <div class="tg-score">${data.credibility_score}%</div>
                </div>
                <span class="tg-badge ${badgeClass}">${data.classification}</span>
            </div>
            
            <div class="tg-body">
                ${data.explanation}
            </div>

            <div class="tg-footer">
                <svg width="10" height="10" viewBox="0 0 24 24" fill="#3b82f6"><path d="M12 2L1 21h22L12 2zm0 3.99L19.53 19H4.47L12 5.99zM13 16h-2v2h2v-2zm0-6h-2v4h2v-4z"/></svg>
                Truth Guardian AI
            </div>
        </div>
        <span class="truth-guardian-icon">
             <svg viewBox="0 0 24 24" width="16" height="16" fill="${iconColor}">
                <path d="M12 1L3 5v6c0 5.55 3.84 10.74 9 12 5.16-1.26 9-6.45 9-12V5l-9-4zm-2 16l-4-4 1.41-1.41L10 14.17l6.59-6.59L18 9l-8 8z"/>
            </svg>
        </span>
    `;

  element.innerHTML += tooltipHTML;
}

// Function to auto-scan paragraphs (disabled by default to be less intrusive, user can rely on context menu)
function scanPage() {
  const paragraphs = document.querySelectorAll('p');
  paragraphs.forEach(p => {
    const text = p.innerText.trim();
    // Heuristic: Only analyze juicy paragraphs
    if (text.length > 100 && text.length < 800 && !p.dataset.tgAnalyzed) {
      // We could inject a small icon at the end of paragraphs to "Analyze this"
      // but for now we'll stick to manual (Context Menu) or Popup for premium feel without spam.
    }
  });
}
