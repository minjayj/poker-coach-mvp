const $ = (id) => document.getElementById(id);
const ui = Object.fromEntries(["refreshSessions", "sessions", "sessionTitle", "sessionMeta", "recordedVideo", "frameSelect", "previousFrame", "nextFrame", "frameImage", "prediction", "labelForm", "heroCards", "boardCards", "actionSummary", "verified", "labelStatus", "trainButton", "trainingReport", "eventList"].map((id) => [id, $(id)]));
const state = { session: null, sessions: [], frames: [], index: -1 };

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `Request failed (${response.status})`);
  return body;
}

async function loadSessions() {
  try {
    state.sessions = await api("/api/play/sessions");
    ui.sessions.replaceChildren();
    if (!state.sessions.length) { ui.sessions.textContent = "No local recordings yet. Start one on the live page."; return; }
    for (const session of state.sessions) {
      const button = document.createElement("button");
      button.textContent = `${new Date(session.started_at).toLocaleString()} · ${session.status} · ${session.frame_count} frames`;
      button.classList.toggle("selected", session.id === state.session?.id);
      button.addEventListener("click", () => selectSession(session));
      ui.sessions.append(button);
    }
    if (!state.session) await selectSession(state.sessions[0]);
  } catch (error) { ui.sessions.textContent = error.message; }
}

async function selectSession(session) {
  state.session = session; state.index = -1;
  ui.sessionTitle.textContent = `Session ${session.id.slice(0, 8)}`;
  ui.sessionMeta.textContent = `${session.status} · ${Math.round(session.video_bytes / 1_000_000)} MB · ${session.frame_count} frames`;
  ui.recordedVideo.hidden = !session.next_chunk;
  if (session.next_chunk) ui.recordedVideo.src = `/api/play/${session.id}/video`;
  try {
    const [frames, events] = await Promise.all([api(`/api/play/${session.id}/frames`), api(`/api/play/${session.id}/events`)]);
    state.frames = frames;
    ui.frameSelect.replaceChildren();
    frames.forEach((frame, index) => {
      const option = document.createElement("option"); option.value = String(index);
      option.textContent = `${frame.id} · ${new Date(frame.captured_at).toLocaleTimeString()}${frame.label ? " · verified" : ""}`;
      ui.frameSelect.append(option);
    });
    ui.frameSelect.disabled = !frames.length;
    if (frames.length) showFrame(0); else { ui.frameImage.hidden = true; ui.prediction.textContent = "No sampled frames in this recording."; }
    ui.eventList.replaceChildren();
    if (!events.length) ui.eventList.textContent = "No confirmed action events were entered during this recording.";
    for (const event of events.slice(-30).reverse()) {
      const line = document.createElement("div"); line.className = "event";
      line.textContent = `${new Date(event.at).toLocaleTimeString()} · ${event.kind} · ${event.payload.player || event.payload.winner || ""} ${event.payload.action || ""} ${event.payload.amount ?? ""}`.trim();
      ui.eventList.append(line);
    }
    [...ui.sessions.children].forEach((button, index) => button.classList.toggle("selected", state.sessions[index]?.id === session.id));
  } catch (error) { ui.prediction.textContent = error.message; }
}

function showFrame(index) {
  if (!state.session || index < 0 || index >= state.frames.length) return;
  state.index = index; ui.frameSelect.value = String(index);
  const frame = state.frames[index];
  ui.frameImage.src = `/api/play/${state.session.id}/frames/${frame.id}/image`;
  ui.frameImage.hidden = false;
  ui.previousFrame.disabled = index === 0; ui.nextFrame.disabled = index === state.frames.length - 1;
  const guess = frame.prediction || {};
  ui.prediction.textContent = `Reader: ${guess.status || "no read"} · Hero ${guess.hero_cards?.join(" ") || "unknown"} · Board ${guess.board_cards?.join(" ") || "unknown"} · Pot ${guess.current_pot ?? "unknown"}. Verify the image yourself.`;
  ui.heroCards.value = (frame.label?.hero_cards || guess.hero_cards || []).join(" ");
  ui.boardCards.value = (frame.label?.board_cards || guess.board_cards || []).join(" ");
  ui.actionSummary.value = frame.label?.action_summary || "";
  ui.verified.checked = false;
  ui.labelStatus.textContent = frame.label ? "Previously verified · edit and save to update" : "Not verified";
}

ui.labelForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.session || state.index < 0) return;
  if (!ui.verified.checked) { ui.labelStatus.textContent = "Check the image and tick verification first."; return; }
  const frame = state.frames[state.index];
  const cards = (value) => value.trim() ? value.trim().split(/[\s,]+/) : [];
  try {
    const saved = await api(`/api/play/${state.session.id}/frames/${frame.id}/label`, { method: "POST", body: JSON.stringify({ verified: true, hero_cards: cards(ui.heroCards.value), board_cards: cards(ui.boardCards.value), action_summary: ui.actionSummary.value }) });
    state.frames[state.index] = saved;
    ui.labelStatus.textContent = "Verified label saved locally";
    ui.frameSelect.options[state.index].textContent += ui.frameSelect.options[state.index].textContent.includes("verified") ? "" : " · verified";
    ui.verified.checked = false;
  } catch (error) { ui.labelStatus.textContent = error.message; }
});

ui.trainButton.addEventListener("click", async () => {
  ui.trainButton.disabled = true; ui.trainingReport.textContent = "Testing on held-out reviewed frames…";
  try {
    const report = await api("/api/play/train", { method: "POST", body: "{}" });
    ui.trainingReport.textContent = `${report.activated ? "New suit calibration activated." : "Current reader kept."}\n${report.reason || ""}\nReviewed frames: ${report.reviewed_frames}; different card patterns: ${report.unique_card_patterns}\nValidation cards: ${report.validation_cards ?? 0}\nBaseline correct: ${report.baseline_correct ?? "—"}; candidate correct: ${report.candidate_correct ?? "—"}\nCandidate errors: ${report.candidate_errors ?? "—"}`;
  } catch (error) { ui.trainingReport.textContent = error.message; }
  finally { ui.trainButton.disabled = false; }
});
ui.frameSelect.addEventListener("change", () => showFrame(Number(ui.frameSelect.value)));
ui.previousFrame.addEventListener("click", () => showFrame(state.index - 1));
ui.nextFrame.addEventListener("click", () => showFrame(state.index + 1));
ui.refreshSessions.addEventListener("click", loadSessions);
loadSessions();
