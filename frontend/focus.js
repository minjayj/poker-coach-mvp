const $ = (id) => document.getElementById(id);
const ui = Object.fromEntries(["connection", "liveBadge", "tableVideo", "videoPlaceholder", "captureCanvas", "shareButton", "stopButton", "recordButton", "captureNote", "variantChip", "formatChip", "handChip", "knownChip", "coachStatus", "messages", "chatForm", "chatInput", "sendButton", "voiceButton", "privacyNote", "azureForm", "azureKey", "azureButton", "azureHint", "setupToggle", "setupOverlay", "setupClose", "setupFrame"].map((id) => [id, $(id)]));
const app = { game: null, context: {}, stream: null, sourceId: null, timer: null, frameBusy: false, session: 0, busy: false, chatQueue: [], voiceOn: false, autoSignature: null, lastVisionPrompt: null, lastPlayerPrompt: null, lastIdentityPrompt: null, lastProfilePrompt: null, lastVisionIssue: null, lastFreezeWarning: false, recording: null, idleStatus: "Ready to think through a spot" };
const STORAGE_KEY = "poker-coach-focus-chat-v1";

async function api(path, options = {}) {
  let response;
  try {
    response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  } catch (error) {
    if (error instanceof TypeError) throw new Error("Local coach server is offline. Restart the Python server, then retry; your message is still in the box.");
    throw error;
  }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(typeof body.detail === "string" ? body.detail : `Request failed (${response.status})`);
  return body;
}

function saveConversation() {
  try {
    const items = [...ui.messages.querySelectorAll(".message")].slice(-40).map((node) => ({ role: node.classList.contains("user") ? "user" : "coach", text: node.querySelector(".bubble").textContent }));
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ items, context: app.context }));
  } catch (_) { /* Browser storage is optional. */ }
}

function recentHistory() {
  return [...ui.messages.querySelectorAll(".message")].slice(-8).map((node) => ({
    role: node.classList.contains("user") ? "user" : "coach",
    text: node.querySelector(".bubble").textContent.slice(0, 600),
  }));
}

function addMessage(role, content) {
  const row = document.createElement("div"); row.className = `message ${role}`;
  if (role === "coach") { const avatar = document.createElement("div"); avatar.className = "mini-avatar"; avatar.textContent = "♠"; row.append(avatar); }
  const wrap = document.createElement("div");
  const bubble = document.createElement("div"); bubble.className = "bubble"; bubble.textContent = content;
  const time = document.createElement("time"); time.textContent = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  wrap.append(bubble, time); row.append(wrap); ui.messages.append(row); ui.messages.scrollTop = ui.messages.scrollHeight;
  saveConversation();
  if (role === "coach" && app.voiceOn && "speechSynthesis" in window) {
    speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(content); utterance.rate = 1.05;
    speechSynthesis.speak(utterance);
  }
}

function restoreConversation() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || "null");
    if (saved?.items?.length) { app.context = saved.context || {}; saved.items.slice(-40).forEach((item) => addMessage(item.role === "user" ? "user" : "coach", item.text)); return; }
  } catch (_) { /* Start clean on malformed storage. */ }
  addMessage("coach", "Share one ClubGG Hold'em table. If window capture freezes, share your display with only one table visible. I'll read cards and pot when clear, then ask about missing actions or amounts.");
}

function renderModelStatus(status) {
  const needsKey = status.provider === "azure" && (!status.available || status.reason.startsWith("Azure request failed"));
  ui.azureForm.hidden = !needsKey;
  app.idleStatus = needsKey && status.available ? "Azure connection failed · re-enter key" :
    status.available ? `${status.model} · ${status.verified ? "conversation ready" : "configured"}` : "Rule-based coaching · connect Azure for conversation";
  ui.privacyNote.textContent = status.provider === "azure" ?
    (status.available ? "Azure chat sends recent messages and saved hand context, not screen video. Poker decisions remain with the local engine." : "Azure chat is off until you connect a key. Poker decisions remain with the local engine.") :
    "Local Ollama chat stays on this computer. Poker decisions remain with the local engine.";
  if (needsKey && status.available) ui.azureHint.textContent = status.reason;
  if (!app.busy) ui.coachStatus.textContent = app.idleStatus;
}

function renderGame(game) {
  const previousCards = app.game?.hero_cards?.join(" ");
  app.game = game;
  ui.variantChip.textContent = game.game_variant === "plo5" ? "PLO5" : "NLH";
  ui.formatChip.textContent = game.game_format === "tournament" ? "Tournament" : "Cash";
  ui.handChip.textContent = `Hand ${game.hand_id}`;
  const cards = game.hero_cards?.length ? game.hero_cards.join(" ") : "cards unknown";
  const pot = game.observed_pot ?? (game.pot > 0 ? game.pot : null);
  const inHand = game.observed_active_players?.length;
  ui.knownChip.textContent = `${cards} · ${pot == null ? "pot unknown" : `pot ${Number(pot).toFixed(2)}`}${inHand ? ` · ${inHand} with cards` : ""}`;
  document.querySelectorAll("[data-variant]").forEach((button) => button.classList.toggle("selected", button.dataset.variant === game.game_variant));
  document.querySelectorAll("[data-format]").forEach((button) => button.classList.toggle("selected", button.dataset.format === game.game_format));
  const uncertainName = game.review_queue?.find((item) => item.field === "identity_seen_player");
  if (uncertainName && uncertainName.id !== app.lastIdentityPrompt) {
    app.lastIdentityPrompt = uncertainName.id;
    addMessage("coach", `Is ${uncertainName.payload.detected_player} the same person as ${uncertainName.payload.candidate_player}? Please confirm in Setup & review before I merge their histories.`);
  }
  if (app.context.hand_id && app.context.hand_id !== game.hand_id) { app.context = {}; addMessage("coach", "A new hand is recorded. Tell me your new cards and betting price when you want a read."); }
  if (previousCards && game.hero_cards?.length === 2 && previousCards !== game.hero_cards.join(" ")) {
    app.context = {};
    app.autoSignature = null;
  }
  maybeAutoCoach(game);
}

async function maybeAutoCoach(game) {
  // Only a future structured vision adapter can satisfy this gate. A video
  // frame or felt-color guess alone never triggers an unsolicited move call.
  const hero = game.seats?.find((seat) => seat.is_hero)?.player;
  const fresh = Date.now() - Date.parse(game.last_input_at || 0) < 5000;
  if (!app.stream || app.busy || !fresh || !game.last_input_verified_spot || !hero || game.observed_actor !== hero ||
      (game.input_confidence ?? 0) < .85 || !game.observed_pot ||
      ![0, 3, 4, 5].includes(game.board_cards?.length) ||
      game.hero_cards?.length !== (game.game_variant === "plo5" ? 5 : 2) ||
      !game.hero_legal_actions?.length || game.review_queue?.length ||
      !game.last_input_source || ["coach-chat", "manual-context"].includes(game.last_input_source)) return;
  const signature = JSON.stringify([game.hand_id, game.game_variant, game.hero_cards, game.board_cards, game.observed_pot, game.hero_to_call, game.actions?.length]);
  if (signature === app.autoSignature) return;
  app.autoSignature = signature;
  const lastAggressor = [...(game.actions || [])].reverse().find((action) => action.player !== hero && ["bet", "raise", "all_in"].includes(action.action));
  const context = { hand_id: game.hand_id, game_variant: game.game_variant, to_call: game.hero_to_call, opponent_id: lastAggressor?.player || null,
    opponent_line: lastAggressor ? `${lastAggressor.player} ${lastAggressor.action}${lastAggressor.amount == null ? "" : ` to ${lastAggressor.amount}`}` : null };
  app.busy = true; ui.coachStatus.textContent = "Reviewing verified table update…";
  try {
    const result = await api("/api/coach/chat", { method: "POST", body: JSON.stringify({ message: "Please analyze this verified live spot", context }) });
    app.context = result.context;
    addMessage("coach", `Live update: ${result.reply}`);
  } catch (error) { addMessage("coach", `I couldn't verify this live spot: ${error.message}`); }
  finally { app.busy = false; ui.coachStatus.textContent = app.idleStatus; drainChatQueue(); }
}

function drainChatQueue() {
  if (!app.busy && app.chatQueue.length) sendChat(app.chatQueue.shift());
}

async function sendChat(text) {
  app.busy = true; ui.coachStatus.textContent = "Thinking through the spot…";
  const history = recentHistory();
  addMessage("user", text);
  try {
    const result = await api("/api/coach/chat", { method: "POST", body: JSON.stringify({ message: text, context: app.context, history }) });
    app.context = result.context || {};
    renderGame(result.state);
    addMessage("coach", result.reply);
    // A failed first model request should reveal the reconnect control;
    // a successful one should update the header to "conversation ready".
    api("/api/coach/model-status").then(renderModelStatus).catch(() => {});
  } catch (error) { addMessage("coach", `I couldn't analyze that yet: ${error.message}. Please check the details and try again.`); }
  finally { app.busy = false; ui.coachStatus.textContent = app.idleStatus; ui.chatInput.focus(); drainChatQueue(); }
}

async function setFormat(changes) {
  if (!app.game) return;
  const payload = { game_format: changes.game_format || app.game.game_format, game_variant: changes.game_variant || app.game.game_variant, cash: app.game.cash, tournament: app.game.tournament };
  try {
    const next = await api("/api/config/game-format", { method: "POST", body: JSON.stringify(payload) });
    renderGame(next); app.context = {};
    addMessage("coach", `Switched to ${next.game_variant === "plo5" ? "PLO5" : "Hold'em"} ${next.game_format}. I'll use the matching analysis path. Rake, payouts and stacks can be set in Setup & review.`);
  } catch (error) { addMessage("coach", `I couldn't switch the game: ${error.message}. You may need to finish the current hand in Setup & review.`); }
}

async function startRecording() {
  if (!app.stream || !app.sourceId || app.recording) return;
  if (!("MediaRecorder" in window)) { addMessage("coach", "This browser cannot record the window. Sharing still works; try a current Chromium browser for local recording."); return; }
  const mimeType = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"].find((value) => MediaRecorder.isTypeSupported(value));
  if (!mimeType) { addMessage("coach", "This browser has no supported WebM recorder. Sharing still works."); return; }
  ui.recordButton.disabled = true;
  let session = null;
  try {
    session = await api("/api/play/start", { method: "POST", body: JSON.stringify({ source_id: app.sourceId, mime_type: mimeType, consent: true }) });
    const recorder = new MediaRecorder(app.stream, { mimeType, videoBitsPerSecond: 1_500_000 });
    let finish;
    const done = new Promise((resolve) => { finish = resolve; });
    const recording = { id: session.id, recorder, queue: Promise.resolve(), nextChunk: 0, error: null, done };
    recorder.addEventListener("dataavailable", (event) => {
      if (!event.data?.size) return;
      const index = recording.nextChunk++;
      recording.queue = recording.queue.then(async () => {
        if (recording.error) return;
        const response = await fetch(`/api/play/${recording.id}/chunks/${index}`, { method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: event.data });
        if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(body.detail || `Chunk upload failed (${response.status})`); }
      }).catch((error) => { recording.error = error; if (recorder.state !== "inactive") recorder.stop(); });
    });
    recorder.addEventListener("stop", async () => {
      await recording.queue;
      try {
        await api(`/api/play/${recording.id}/stop`, { method: "POST", body: JSON.stringify({ incomplete: !!recording.error }) });
        addMessage("coach", recording.error ? `Recording saved as incomplete: ${recording.error.message}. Review what was captured after the hand.` : "Local recording saved. Review and label frames after the hand; only confirmed labels can train the card reader.");
      } catch (error) { addMessage("coach", `Recording stopped, but final save could not be confirmed: ${error.message}`); }
      if (app.recording === recording) app.recording = null;
      ui.recordButton.textContent = "Start local recording"; ui.recordButton.classList.remove("recording"); ui.recordButton.disabled = false;
      ui.liveBadge.textContent = app.stream ? "Sharing live" : "Not sharing"; ui.liveBadge.classList.remove("recording");
      finish();
    });
    app.recording = recording;
    recorder.start(2000);
    ui.recordButton.textContent = "Stop recording"; ui.recordButton.classList.add("recording");
    ui.liveBadge.textContent = "Recording locally"; ui.liveBadge.classList.add("recording");
    addMessage("coach", "Recording this shared window locally in the play folder. Nothing is uploaded to an online service. Stop it whenever you want.");
  } catch (error) {
    if (session) await api(`/api/play/${session.id}/stop`, { method: "POST", body: JSON.stringify({ incomplete: true }) }).catch(() => {});
    if (app.recording?.id === session?.id) app.recording = null;
    addMessage("coach", `Recording did not start: ${error.message}`);
    ui.recordButton.textContent = "Start local recording"; ui.recordButton.classList.remove("recording"); ui.recordButton.disabled = false;
    ui.liveBadge.textContent = app.stream ? "Sharing live" : "Not sharing"; ui.liveBadge.classList.remove("recording");
  }
}

async function stopRecording() {
  const recording = app.recording;
  if (!recording) return;
  ui.recordButton.disabled = true;
  if (recording.recorder.state !== "inactive") recording.recorder.stop();
  await recording.done;
}

async function stopSharing() {
  app.session += 1;
  if (app.timer) clearInterval(app.timer); app.timer = null;
  await stopRecording();
  const oldSource = app.sourceId; app.sourceId = null;
  if (app.stream) app.stream.getTracks().forEach((track) => track.stop());
  app.stream = null; ui.tableVideo.srcObject = null; ui.tableVideo.hidden = true; ui.videoPlaceholder.hidden = false;
  app.lastVisionPrompt = null; app.lastPlayerPrompt = null; app.lastIdentityPrompt = null; app.lastProfilePrompt = null; app.lastVisionIssue = null; app.lastFreezeWarning = false;
  ui.shareButton.hidden = false; ui.stopButton.hidden = true; ui.recordButton.hidden = true; ui.liveBadge.textContent = "Not sharing"; ui.liveBadge.classList.remove("active", "recording");
  ui.captureNote.textContent = "Local capture · no game controls";
  if (oldSource) await api("/api/input/stop", { method: "POST", body: JSON.stringify({ source_id: oldSource }) }).catch(() => {});
}

async function sendFrame(session, sourceId) {
  if (session !== app.session || app.frameBusy || !ui.tableVideo.videoWidth) return;
  app.frameBusy = true;
  const controller = new AbortController();
  const deadline = setTimeout(() => controller.abort(), 12000);
  try {
    const canvas = ui.captureCanvas;
    canvas.width = Math.min(1600, ui.tableVideo.videoWidth);
    canvas.height = Math.round(ui.tableVideo.videoHeight * canvas.width / ui.tableVideo.videoWidth);
    canvas.getContext("2d").drawImage(ui.tableVideo, 0, 0, canvas.width, canvas.height);
    const result = await api("/api/input/frame", { method: "POST", body: JSON.stringify({ source_id: sourceId, image_data: canvas.toDataURL("image/jpeg", .88) }), signal: controller.signal });
    if (session === app.session) {
      renderGame(result.state);
      const read = result.vision;
      if (read) {
        ui.captureNote.textContent = read.status === "unsupported_layout" ? "Can't isolate one table · share a single game window" :
          read.status === "reader_unavailable" ? "Reader unavailable · confirm details in chat" :
          read.status === "not_holdem" ? "Hold'em reader inactive for this window" :
          read.status === "profile_read" ? `Profile read · ${read.observed_profile?.player || "name unclear"}` :
          `Window reader · ${read.hero_cards?.join(" ") || "cards unclear"} · ${read.current_pot == null ? "pot unclear" : `pot ${read.current_pot.toFixed(2)}`} · ${read.active_names?.length || 0} with cards`;
        const visibleIssues = (read.missing || []).filter((item) => !item.startsWith("Prior betting") && !item.startsWith("Profile captured") && !item.startsWith("The bettor and prior action") && !item.startsWith("The amount to call"));
        const issueKey = JSON.stringify([read.status, visibleIssues]);
        if (visibleIssues.length && issueKey !== app.lastVisionIssue) {
          app.lastVisionIssue = issueKey;
          addMessage("coach", `I can't verify everything in this view. ${visibleIssues.join(" ")} If you're in the hand, please tell me the missing details in chat.`);
        }
        if (read.observed_profile) {
          const profile = read.observed_profile;
          const key = JSON.stringify([profile.player, profile.hands_seen, profile.vpip, profile.pfr]);
          if (key !== app.lastProfilePrompt) {
            app.lastProfilePrompt = key;
            addMessage("coach", `I read ${profile.player}'s displayed profile: ${profile.hands_seen} hands, VPIP ${profile.vpip}%, PFR ${profile.pfr}%. If this name differs from a player you already know, confirm the identity in Setup & review before I merge profiles.`);
          }
        }
        const game = result.state;
        const playersKey = JSON.stringify(game.observed_players || []);
        if (game.observed_players?.length && playersKey !== app.lastPlayerPrompt) {
          app.lastPlayerPrompt = playersKey;
          addMessage("coach", `I can read these players at the table: ${game.observed_players.join(", ")}. I see cards at ${game.observed_active_players?.length || 0} seats; profile stats still require opening each player's profile.`);
        }
        const promptKey = JSON.stringify([game.hero_cards, game.board_cards, game.observed_pot, game.observed_hero_to_call, game.observed_active_players]);
        if (!result.input?.frame_static && game.hero_cards?.length === 2 && game.observed_pot != null && promptKey !== app.lastVisionPrompt) {
          app.lastVisionPrompt = promptKey;
          const table = game.observed_active_players?.length ? `${game.observed_active_players.length} players with cards (${game.observed_active_players.join(", ")}). ` : "The active-player count is unclear. ";
          const price = game.observed_hero_turn && game.observed_hero_to_call != null ? `Your call price is ${Number(game.observed_hero_to_call).toFixed(2)}. ` : "I can't verify your call price yet. ";
          addMessage("coach", `I can read ${game.hero_cards.join(" ")}${game.board_cards?.length ? ` on ${game.board_cards.join(" ")}` : " preflop"} and pot ${Number(game.observed_pot).toFixed(2)}. ${table}${price}I still need to know who made the prior bet or raise before giving a player-specific read.`);
        }
      }
      if (result.input?.frame_static) {
        ui.captureNote.textContent = `Image unchanged for ${Math.floor(result.input.unchanged_seconds)}s · check the shared window`;
        if (!app.lastFreezeWarning) {
          app.lastFreezeWarning = true;
          addMessage("coach", "The shared picture has stopped changing. If the game is moving, keep ClubGG visible and unminimized. Stop sharing and try sharing your display instead; I won't assume this old picture is live.");
        }
      } else if (app.lastFreezeWarning) {
        app.lastFreezeWarning = false;
        addMessage("coach", "The shared picture is updating again. I'll re-check the cards and pot.");
      }
    }
  } catch (error) { if (session === app.session) ui.captureNote.textContent = error.name === "AbortError" ? "Frame analysis timed out · trying the next frame" : `Capture error: ${error.message}`; }
  finally { clearTimeout(deadline); app.frameBusy = false; }
}

async function startSharing() {
  try {
    const stream = await navigator.mediaDevices.getDisplayMedia({ video: { frameRate: { ideal: 2, max: 3 }, width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false });
    app.stream = stream; app.session += 1; const session = app.session;
    const surface = stream.getVideoTracks()[0]?.getSettings()?.displaySurface || "unknown";
    const id = globalThis.crypto?.randomUUID?.() || String(Date.now());
    app.sourceId = `clubgg-${surface}-share-${id}`;
    ui.tableVideo.srcObject = stream; ui.tableVideo.hidden = false; ui.videoPlaceholder.hidden = true;
    await ui.tableVideo.play();
    ui.shareButton.hidden = true; ui.stopButton.hidden = false; ui.liveBadge.textContent = "Sharing live"; ui.liveBadge.classList.add("active");
    ui.captureNote.textContent = surface === "window" ? "Window feed active · checking cards and pot" : "Looking for one ClubGG table in the shared view";
    if (surface === "window") addMessage("coach", "Window share connected. I'm checking for your cards and the pot; I'll ask about any action or amount I can't read.");
    if (surface !== "window") addMessage("coach", "I'll try to isolate one ClubGG table from this share. If multiple tables are visible, I'll ask you to share one game window instead.");
    const track = stream.getVideoTracks()[0]; track.addEventListener("ended", () => { if (session === app.session) stopSharing(); });
    ui.recordButton.hidden = surface !== "window";
    app.timer = setInterval(() => sendFrame(session, app.sourceId), 1200);
    void sendFrame(session, app.sourceId);
  } catch (error) { await stopSharing(); if (error.name !== "NotAllowedError") addMessage("coach", `Sharing didn't start: ${error.message}`); }
}

function connectSocket() {
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  const socket = new WebSocket(`${protocol}//${location.host}/ws`);
  socket.addEventListener("open", () => { ui.connection.textContent = "Connected"; ui.connection.classList.add("online"); });
  socket.addEventListener("message", (event) => renderGame(JSON.parse(event.data)));
  socket.addEventListener("close", () => { ui.connection.textContent = "Reconnecting"; ui.connection.classList.remove("online"); setTimeout(connectSocket, 2000); });
}

function setSetupOpen(open) {
  ui.setupOverlay.hidden = !open;
  ui.setupToggle.setAttribute("aria-expanded", String(open));
  document.body.classList.toggle("setup-open", open);
  document.querySelector(".shell").inert = open;
  if (open) {
    // Load the existing full setup only when requested; live capture stays active.
    if (!ui.setupFrame.src) ui.setupFrame.src = ui.setupFrame.dataset.src;
    ui.setupClose.focus();
  } else {
    ui.setupToggle.focus();
  }
}

ui.chatForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = ui.chatInput.value.trim();
  if (!text) return;
  ui.chatInput.value = "";
  app.chatQueue.push(text);
  if (app.busy) ui.coachStatus.textContent = "Message queued · finishing the current thought…";
  drainChatQueue();
});
ui.sendButton.addEventListener("click", (event) => {
  // Some embedded browsers do not dispatch the form's default submit action
  // from an icon-only button. Use the same path as pressing Enter.
  event.preventDefault();
  ui.chatForm.requestSubmit();
});
ui.azureForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const key = ui.azureKey.value.trim();
  if (!key) return;
  ui.azureButton.disabled = true;
  ui.azureHint.textContent = "Checking Azure connection…";
  try {
    const status = await api("/api/coach/connect-azure", { method: "POST", body: JSON.stringify({ api_key: key }) });
    renderModelStatus(status);
    if (status.verified) addMessage("coach", "Azure conversation is connected. Ask me a question, or share the game window when you're ready.");
    else ui.azureHint.textContent = status.reason;
  } catch (error) { ui.azureHint.textContent = `Connection failed: ${error.message}`; }
  finally { ui.azureKey.value = ""; ui.azureButton.disabled = false; }
});
ui.chatInput.addEventListener("keydown", (event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); ui.chatForm.requestSubmit(); } });
ui.shareButton.addEventListener("click", startSharing);
ui.stopButton.addEventListener("click", stopSharing);
ui.recordButton.addEventListener("click", () => app.recording ? stopRecording() : startRecording());
ui.setupToggle.addEventListener("click", () => setSetupOpen(ui.setupOverlay.hidden));
ui.setupClose.addEventListener("click", () => setSetupOpen(false));
ui.setupOverlay.addEventListener("click", (event) => { if (event.target === ui.setupOverlay) setSetupOpen(false); });
document.addEventListener("keydown", (event) => { if (event.key === "Escape" && !ui.setupOverlay.hidden) setSetupOpen(false); });
ui.voiceButton.addEventListener("click", () => {
  app.voiceOn = !app.voiceOn;
  ui.voiceButton.textContent = app.voiceOn ? "Voice on" : "Voice off";
  ui.voiceButton.setAttribute("aria-label", app.voiceOn ? "Turn coach voice off" : "Turn coach voice on");
  ui.voiceButton.classList.toggle("enabled", app.voiceOn);
  if (!app.voiceOn && "speechSynthesis" in window) speechSynthesis.cancel();
});
document.querySelectorAll("[data-variant]").forEach((button) => button.addEventListener("click", () => setFormat({ game_variant: button.dataset.variant })));
document.querySelectorAll("[data-format]").forEach((button) => button.addEventListener("click", () => setFormat({ game_format: button.dataset.format })));
restoreConversation();
api("/api/state").then(renderGame).catch((error) => addMessage("coach", `Local server unavailable: ${error.message}`));
api("/api/coach/model-status").then(renderModelStatus).catch(() => {});
connectSocket();
