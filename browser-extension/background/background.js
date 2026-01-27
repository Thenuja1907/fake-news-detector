// Background Service Worker

// 1. Create Context Menu on Install
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "analyze-text",
    title: "🛡️ Analyze Verification for selection",
    contexts: ["selection"]
  });
  console.log("Truth Guardian: Installed");
});

// 2. Handle Context Menu Clicks
chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "analyze-text" && info.selectionText) {
    // Send a message to the content script to display a loader/result
    // OR open the popup. Opening popup programmatically isn't possible directly.
    // We will inject a result overlay into the content page.

    chrome.tabs.sendMessage(tab.id, {
      action: "analyzeSelection",
      text: info.selectionText
    });
  }
});

// 3. Listen for requests from Content Script (if it needs to proxy through background, typically not needed for fetch if CORS is open, but good for state)
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "checkStatus") {
    sendResponse({ status: "active" });
  }
});
