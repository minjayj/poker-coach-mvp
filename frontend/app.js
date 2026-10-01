const state = { game: null, players: [], mediaStream: null, captureTimer: null, captureInFlight: false, captureSourceId: null, captureSession: 0, coachBrief: null };

const elements = Object.fromEntries([
  "socketStatus", "messageText", "potValue", "streetValue", "currentActionValue", "playersInHandValue", "decisionContextValue", "reviewList",
  "liveVideo", "remoteFramePreview", "videoEmpty", "captureCanvas", "inputStatus", "captureGuide",
  "coachHeadline", "coachSee", "coachWhy", "coachWatch", "coachRep", "coachOpponent", "coachToCall", "coachRaiseTo",
  "heroCardsInput", "boardCardsInput", "observedPotInput", "visionStatusText", "observedStatsPlayer", "observedHandsInput", "observedVpipInput", "observedPfrInput", "observedAfInput",
  "gameFormatInput", "gameVariantInput", "formatStatus", "cashFields", "rakePercentInput", "rakeCapInput", "noFlopNoDropInput", "tournamentFields", "playersRemainingInput", "paidPlacesInput", "anteInput", "finalTableSizeInput", "payoutsInput", "profileDetail",
  "seatRows", "smallBlindInput", "bigBlindInput", "dealerSeatInput", "actionPlayer", "actionType", "actionAmount", "amountType", "streetInput", "transcriptInput",
  "seatsList", "profilesList", "actionLog",
  "studyHeroCards", "studyStackBb", "studyDecision", "studyResult",
].map((id) => [id, document.querySelector(`#${id}`)]));

function showMessage(text) { elements.messageText.textContent = text; }
function formatMoney(amount) { return `$${Number(amount || 0).toFixed(2)}`; }
function parseCardList(value) { return value.split(",").map((card) => card.trim()).filter(Boolean); }
function optionalNumber(element) { return element.value.trim() === "" ? null : Number(element.value); }
function escapeHtml(value) { return String(value ?? "").replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[char])); }

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  if (!response.ok) throw new Error((await response.text()) || `Request failed: ${response.status}`);
  return response.json();
}

function createSeatRow(seat = { seat: elements.seatRows.children.length + 1, player: "", stack: 100, isHero: false }) {
  const row = document.createElement("div"); row.className = "seat-row";
  row.innerHTML = `<label>Seat<input type="number" class="seat-number" min="1" value="${seat.seat}" /></label><label>Player<input type="text" class="seat-player" placeholder="Player name" value="${escapeHtml(seat.player)}" /></label><label>Stack<input type="number" class="seat-stack" min="0" step="0.5" value="${seat.stack}" /></label><label class="hero-toggle"><input type="radio" name="heroSeat" class="seat-hero" ${seat.isHero ? "checked" : ""} />Hero</label><button type="button" class="ghost-button seat-remove">Remove</button>`;
  row.querySelector(".seat-remove").addEventListener("click", () => row.remove()); elements.seatRows.appendChild(row);
}

function collectSeatConfig() {
  const seats = [...elements.seatRows.querySelectorAll(".seat-row")].map((row) => ({ seat: Number(row.querySelector(".seat-number").value), player: row.querySelector(".seat-player").value.trim(), stack: Number(row.querySelector(".seat-stack").value || 0), is_hero: row.querySelector(".seat-hero").checked })).filter((seat) => seat.player);
  return { seats, hero_seat: seats.find((seat) => seat.is_hero)?.seat ?? null, dealer_seat: Number(elements.dealerSeatInput.value || 1), blinds: { small: Number(elements.smallBlindInput.value || 1), big: Number(elements.bigBlindInput.value || 2) } };
}

function setOptions(select, values, emptyText) {
  const selected = select.value; select.innerHTML = "";
  if (!values.length) { select.innerHTML = `<option value="">${emptyText}</option>`; return; }
  values.forEach((value) => { const option = document.createElement("option"); option.value = value; option.textContent = value; select.appendChild(option); });
  select.value = values.includes(selected) ? selected : values[0];
}

function updateSelects() {
  const players = state.game?.players || []; setOptions(elements.actionPlayer, players, "Configure seats first");
  const opponents = (state.game?.seats || []).filter((seat) => !seat.is_hero).map((seat) => seat.player);
  setOptions(elements.coachOpponent, opponents, "Configure an opponent first");
  setOptions(elements.observedStatsPlayer, opponents, "Configure opponents first");
}

function renderSeats() {
  const seats = state.game?.seats || [];
  elements.seatsList.innerHTML = seats.length ? seats.map((seat) => `<article class="seat-card"><p class="label">Seat ${seat.seat}${seat.is_hero ? " · Hero" : ""}</p><strong>${escapeHtml(seat.player)}</strong><span>${formatMoney(state.game?.stacks?.[seat.player] ?? seat.stack)} behind</span></article>`).join("") : `<p class="subtle">No seats configured yet.</p>`;
}

function renderProfiles() {
  elements.profilesList.innerHTML = state.players.length ? state.players.map((profile) => `<button type="button" class="profile-card" data-player="${escapeHtml(profile.name)}"><h3>${escapeHtml(profile.name)}</h3><p>${profile.hands_seen} hands · VPIP ${Math.round((profile.vpip || 0) * 100)}% · PFR ${Math.round((profile.pfr || 0) * 100)}%</p><p>Calls ${profile.call_count} · Raises ${profile.raise_count} · AF ${Number(profile.aggression_factor || 0).toFixed(2)}</p><p>Folds ${profile.folds} · Bets ${profile.bets} · Checks ${profile.checks} · Wins ${profile.win_count}</p></button>`).join("") : `<p class="subtle">Profiles appear after seats are saved; completed hands build opponent memory.</p>`;
  elements.profilesList.querySelectorAll("[data-player]").forEach((button) => button.addEventListener("click", () => { elements.coachOpponent.value = button.dataset.player; showMessage(`${button.dataset.player} selected for the next coaching request.`); }));
  elements.profilesList.querySelectorAll("[data-player]").forEach((button) => button.addEventListener("click", async () => {
    try {
      const result = await api(`/api/players/${encodeURIComponent(button.dataset.player)}/phase-breakdown`);
      elements.profileDetail.innerHTML = result.phases.length
        ? `<strong>${escapeHtml(button.dataset.player)} · ${escapeHtml((state.game?.game_variant || "nlh").toUpperCase())} by stage</strong>${result.phases.map((phase) => `<p>${escapeHtml(phase.tournament_phase.replaceAll("_", " "))}: ${phase.hands_seen} local hands · VPIP ${Math.round(100 * phase.vpip_count / phase.hands_seen)}% · PFR ${Math.round(100 * phase.pfr_count / phase.hands_seen)}% · calls ${phase.call_count} · raises ${phase.raise_count} · folds ${phase.fold_count}</p>`).join("")}`
        : `No completed local hands for ${escapeHtml(button.dataset.player)} yet. Displayed profile stats remain available above.`;
    } catch (error) { elements.profileDetail.textContent = `Could not load stage history: ${error.message}`; }
  }));
  state.players.filter((profile) => profile.observed_hands_seen != null).forEach((profile) => {
    const card = [...elements.profilesList.querySelectorAll("[data-player]")].find((button) => button.dataset.player === profile.name);
    if (!card) return;
    const source = profile.observed_source?.startsWith("clubgg_profile:") ? "ClubGG baseline" : "Observed baseline";
    card.insertAdjacentHTML("beforeend", `<p>${source}: ${profile.observed_hands_seen} hands Â· VPIP ${Math.round((profile.observed_vpip || 0) * 100)}% Â· PFR ${Math.round((profile.observed_pfr || 0) * 100)}% Â· AF ${Number(profile.observed_aggression_factor || 0).toFixed(2)}</p>`);
  });
}

function renderActionLog() {
  const actions = state.game?.actions || [];
  elements.actionLog.innerHTML = actions.length ? actions.slice().reverse().map((action) => `<article class="log-item"><strong>${escapeHtml(action.player || "Unknown player")} ${escapeHtml(action.action.replace("_", " "))}</strong><span>${action.amount != null ? formatMoney(action.amount) : "No amount"} · ${action.street} · ${action.source}</span></article>`).join("") : `<p class="subtle">No actions logged yet.</p>`;
}

function renderReviewQueue() {
  const reviews = state.game?.review_queue || [];
  const describe = (review) => {
    const payload = review.payload || {};
    if (review.field === "identity_action") return `Is “${payload.detected_player}” the same player as “${payload.candidate_player}”? Detected: ${payload.action}${payload.amount != null ? ` ${formatMoney(payload.amount)}` : ""}.`;
    if (review.field === "identity_actor") return `Is “${payload.detected_player}” the same player as “${payload.candidate_player}”?`;
    if (review.field === "ambiguous_action") return `Could not safely identify this action: ${payload.reason || "missing information"}. Detected action: ${payload.action || "unknown"}.`;
    if (review.field === "action" && payload.reason === "missing_amount") return `Detected a ${payload.action}, but the amount is missing. Enter it manually before using the hand state.`;
    return `${review.field}: ${JSON.stringify(payload)}`;
  };
  elements.reviewList.innerHTML = reviews.length ? reviews.map((review) => `<article class="log-item"><strong>Review needed · ${Math.round(Number(review.confidence || 0) * 100)}% confidence</strong><span>${escapeHtml(describe(review))}</span><div class="button-row"><button type="button" data-review="${escapeHtml(review.id)}" data-decision="approve">Apply</button><button type="button" class="ghost-button" data-review="${escapeHtml(review.id)}" data-decision="dismiss">Dismiss</button></div></article>`).join("") : `<p class="subtle">No items need review.</p>`;
  elements.reviewList.querySelectorAll("[data-review]").forEach((button) => button.addEventListener("click", async () => {
    try {
      state.game = await api(`/api/reviews/${button.dataset.review}`, { method: "POST", body: JSON.stringify({ decision: button.dataset.decision }) });
      renderSummary(); renderReviewQueue(); showMessage(`Vision item ${button.dataset.decision === "approve" ? "applied" : "dismissed"}.`);
    } catch (error) { showMessage(`Could not apply vision item: ${error.message}`); }
  }));
}

function renderSummary() {
  const game = state.game; if (!game) return;
  elements.potValue.textContent = formatMoney(game.observed_pot ?? game.pot); elements.streetValue.textContent = game.current_street;
  elements.currentActionValue.textContent = game.current_action; elements.playersInHandValue.textContent = (game.active_players?.length || game.observed_active_players?.length || game.players?.length || 0);
  const variant = (game.game_variant || "nlh").toUpperCase();
  if (document.activeElement !== elements.gameVariantInput) elements.gameVariantInput.value = game.game_variant || "nlh";
  elements.heroCardsInput.placeholder = game.game_variant === "plo5" ? "Ah, Kh, Qh, Jd, Tc" : "Ah, Kh";
  elements.formatStatus.textContent = game.game_format === "tournament" ? `${variant} tournament · ${game.tournament_phase.replaceAll("_", " ")}` : `${variant} cash game`;
  const legal = (game.hero_legal_actions || []).join(", ") || "not active";
  elements.decisionContextValue.textContent = `Hero to call: ${formatMoney(game.hero_to_call)} · legal: ${legal} · position: ${game.positions?.[game.seats?.find((seat) => seat.is_hero)?.player] || "unknown"}`;
  elements.heroCardsInput.value = (game.hero_cards || []).join(", "); elements.boardCardsInput.value = (game.board_cards || []).join(", ");
  elements.observedPotInput.value = game.observed_pot ?? ""; elements.dealerSeatInput.value = game.dealer_seat ?? 1; updateSelects(); renderSeats(); renderActionLog(); renderReviewQueue();
}

function toggleTournamentFields() {
  elements.tournamentFields.hidden = elements.gameFormatInput.value !== "tournament";
  elements.cashFields.hidden = elements.gameFormatInput.value !== "cash";
}

function syncFormatForm() {
  const game = state.game;
  if (!game) return;
  elements.gameFormatInput.value = game.game_format || "cash";
  elements.gameVariantInput.value = game.game_variant || "nlh";
  elements.rakePercentInput.value = game.cash?.rake_percent ?? 0;
  elements.rakeCapInput.value = game.cash?.rake_cap ?? "";
  elements.noFlopNoDropInput.checked = game.cash?.no_flop_no_drop ?? true;
  elements.playersRemainingInput.value = game.tournament?.players_remaining ?? "";
  elements.paidPlacesInput.value = game.tournament?.paid_places ?? "";
  elements.anteInput.value = game.tournament?.ante ?? 0;
  elements.finalTableSizeInput.value = game.tournament?.final_table_size ?? 9;
  elements.payoutsInput.value = (game.tournament?.payouts || []).join(", ");
  toggleTournamentFields();
}

async function saveGameFormat() {
  const payoutText = elements.payoutsInput.value.trim();
  const payouts = payoutText ? payoutText.split(",").map((value) => Number(value.trim())) : [];
  if (payouts.some((value) => !Number.isFinite(value) || value < 0)) return showMessage("Enter prizes as nonnegative numbers separated by commas.");
  const tournament = {
    players_remaining: optionalNumber(elements.playersRemainingInput),
    paid_places: optionalNumber(elements.paidPlacesInput),
    payouts,
    ante: Number(elements.anteInput.value || 0),
    final_table_size: Number(elements.finalTableSizeInput.value || 9),
  };
  const cash = { rake_percent: Number(elements.rakePercentInput.value || 0), rake_cap: optionalNumber(elements.rakeCapInput), no_flop_no_drop: elements.noFlopNoDropInput.checked };
  try {
    state.game = await api("/api/config/game-format", { method: "POST", body: JSON.stringify({ game_format: elements.gameFormatInput.value, game_variant: elements.gameVariantInput.value, cash, tournament }) });
    renderSummary(); syncFormatForm(); await refreshPlayers(); showMessage("Game format, variant, and payout context saved.");
  } catch (error) { showMessage(`Could not save game format: ${error.message}`); }
}

function renderCoach(brief) {
  state.coachBrief = brief; elements.coachHeadline.textContent = brief.headline;
  const list = (target, items) => { target.innerHTML = ""; items.forEach((item) => { const li = document.createElement("li"); li.textContent = item; target.appendChild(li); }); };
  list(elements.coachSee, brief.what_i_see || []); list(elements.coachWhy, brief.why || []); list(elements.coachWatch, brief.watch_out_for || []); elements.coachRep.textContent = brief.next_rep || "";
}

async function refreshPlayers() { state.players = await api("/api/players"); renderProfiles(); }
async function refreshState() { state.game = await api("/api/state"); renderSummary(); }

async function refreshInputPreview() {
  const input = await api("/api/input/latest");
  if (state.mediaStream) return;
  const age = input.last_received_at ? Date.now() - Date.parse(input.last_received_at) : Infinity;
  elements.inputStatus.textContent = input.source_id ? `${input.source_id} · ${age < 5000 ? "receiving" : "stale"}` : "No source";
  if (input.image_data) { elements.remoteFramePreview.src = input.image_data; elements.remoteFramePreview.classList.add("visible"); elements.videoEmpty.hidden = true; }
  else { elements.remoteFramePreview.removeAttribute("src"); elements.remoteFramePreview.classList.remove("visible"); elements.videoEmpty.hidden = false; }
}

function stopInput() {
  state.captureSession += 1;
  const oldSource = state.captureSourceId;
  if (state.captureTimer) window.clearInterval(state.captureTimer); state.captureTimer = null;
  state.mediaStream?.getTracks().forEach((track) => track.stop()); state.mediaStream = null; elements.liveVideo.srcObject = null;
  state.captureSourceId = null; state.captureInFlight = false;
  elements.liveVideo.classList.remove("visible"); elements.inputStatus.textContent = "Input stopped";
  elements.remoteFramePreview.classList.remove("visible"); elements.videoEmpty.hidden = false;
  elements.captureGuide.textContent = "Select the ClubGG window in the picker whenever you are ready to resume local capture.";
  if (oldSource) api("/api/input/stop", { method: "POST", body: JSON.stringify({ source_id: oldSource }) }).catch(() => {});
}

async function sendFrame(sourceId, session) {
  if (session !== state.captureSession || state.captureInFlight || !state.mediaStream || !elements.liveVideo.videoWidth) return;
  state.captureInFlight = true;
  try {
    const canvas = elements.captureCanvas; const width = Math.min(1600, elements.liveVideo.videoWidth); const height = Math.round(elements.liveVideo.videoHeight * (width / elements.liveVideo.videoWidth));
    canvas.width = width; canvas.height = height; canvas.getContext("2d").drawImage(elements.liveVideo, 0, 0, width, height);
    const result = await api("/api/input/frame", { method: "POST", body: JSON.stringify({ source_id: sourceId, image_data: canvas.toDataURL("image/jpeg", 0.88) }) });
    if (session === state.captureSession) elements.inputStatus.textContent = `${sourceId} · ${result.vision?.status || "streaming"}`;
  } catch (error) { if (session === state.captureSession) { elements.inputStatus.textContent = "Input error"; showMessage(`Frame upload failed: ${error.message}`); } }
  finally { if (session === state.captureSession) state.captureInFlight = false; }
}

async function startInput(mode) {
  stopInput();
  const session = state.captureSession;
  try {
    const stream = mode === "screen"
      ? await navigator.mediaDevices.getDisplayMedia({ video: { frameRate: { ideal: 2, max: 3 }, width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false })
      : await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false });
    if (session !== state.captureSession) { stream.getTracks().forEach((track) => track.stop()); return; }
    state.mediaStream = stream;
    elements.liveVideo.srcObject = state.mediaStream; await elements.liveVideo.play(); elements.liveVideo.classList.add("visible"); elements.remoteFramePreview.classList.remove("visible"); elements.videoEmpty.hidden = true;
    const track = state.mediaStream.getVideoTracks()[0];
    if ("contentHint" in track) track.contentHint = "detail";
    const displaySurface = track.getSettings?.().displaySurface || "shared";
    const captureId = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    const sourceId = mode === "screen" ? `clubgg-${displaySurface}-share-${captureId}` : `browser-camera-${captureId}`;
    state.captureSourceId = sourceId;
    if (mode === "screen") {
      const selectedWindow = displaySurface === "window";
      elements.captureGuide.textContent = selectedWindow
        ? "ClubGG window capture is active locally. Frames are compressed and sent about once per second; stop sharing any time."
        : "Capture is active, but this is not a single application window. For the cleanest input, stop and select the ClubGG window in the picker.";
    } else {
      elements.captureGuide.textContent = "Camera capture is active locally. A future vision adapter can submit structured observations alongside these preview frames.";
    }
    await sendFrame(sourceId, session);
    if (session !== state.captureSession) return;
    state.captureTimer = window.setInterval(() => sendFrame(sourceId, session), 1200);
    track.addEventListener("mute", () => { if (session === state.captureSession) elements.inputStatus.textContent = "Capture paused"; });
    track.addEventListener("unmute", () => { if (session === state.captureSession) elements.inputStatus.textContent = `${sourceId} streaming`; });
    track.addEventListener("ended", () => { if (session === state.captureSession) stopInput(); });
    showMessage(`${mode === "screen" ? "ClubGG window share" : "Camera"} connected. Preview frames are flowing to the local app.`);
  } catch (error) { if (session === state.captureSession) { stopInput(); showMessage(`Could not start ${mode}: ${error.message}`); } }
}

async function requestCoach() {
  elements.coachHeadline.textContent = "Coach is reviewing the current hand…";
  try {
    const payload = { opponent_id: elements.coachOpponent.value || null, to_call: elements.coachToCall.value ? Number(elements.coachToCall.value) : null, raise_to: elements.coachRaiseTo.value ? Number(elements.coachRaiseTo.value) : null };
    const brief = await api("/api/coach/brief", { method: "POST", body: JSON.stringify(payload) }); renderCoach(brief); showMessage("Coaching notes are ready. Review the reasoning before acting.");
  } catch (error) { elements.coachHeadline.textContent = "Coach could not analyze this spot."; showMessage(`Coach error: ${error.message}`); }
}

async function loadOpenStudyCatalog() {
  try {
    const catalog = await api("/api/study/push-fold");
    elements.studyStackBb.innerHTML = "";
    catalog.scenarios.forEach((scenario) => {
      const option = document.createElement("option");
      option.value = String(scenario.effective_stack_bb);
      option.textContent = `${scenario.effective_stack_bb}bb · ${scenario.quality}`;
      elements.studyStackBb.appendChild(option);
    });
  } catch (error) {
    elements.studyResult.textContent = `Open study data is unavailable: ${error.message}`;
    document.querySelector("#lookupStudyButton").disabled = true;
  }
}

async function lookupOpenStudy() {
  const cards = parseCardList(elements.studyHeroCards.value);
  if (cards.length !== 2) {
    elements.studyResult.textContent = "Enter exactly two cards, such as Ah, Kh.";
    return;
  }
  try {
    const result = await api("/api/study/push-fold", {
      method: "POST",
      body: JSON.stringify({ hero_hand: cards, effective_stack_bb: Number(elements.studyStackBb.value), decision: elements.studyDecision.value }),
    });
    const frequencies = Object.entries(result.action_frequencies).map(([action, frequency]) => `${action} ${(frequency * 100).toFixed(1)}%`).join(" · ");
    const values = Object.entries(result.model_action_values_bb).map(([action, value]) => `${action} ${Number(value).toFixed(2)}bb`).join(" · ");
    elements.studyResult.textContent = `${result.hand_class} at ${result.effective_stack_bb}bb — ${frequencies}. Model action values: ${values}. Quality: ${result.quality}; model-game exploitability ${Number(result.model_exploitability_bb).toFixed(4)}bb; action-value error bound ${Number(result.action_value_standard_error_upper_bound_bb).toFixed(4)}bb. ${result.assumptions.join(" ")} Source: ${result.source_url}`;
  } catch (error) {
    elements.studyResult.textContent = `No matching study spot: ${error.message}`;
  }
}

function speakCoach() {
  if (!("speechSynthesis" in window)) { showMessage("This browser does not support reading notes aloud."); return; }
  const brief = state.coachBrief; if (!brief) { showMessage("Request coaching before reading notes aloud."); return; }
  speechSynthesis.cancel(); speechSynthesis.speak(new SpeechSynthesisUtterance([brief.headline, ...(brief.why || []), ...(brief.watch_out_for || [])].join(" ")));
}

document.querySelector("#addSeatButton").addEventListener("click", () => createSeatRow());
document.querySelector("#startCameraButton").addEventListener("click", () => startInput("camera")); document.querySelector("#shareScreenButton").addEventListener("click", () => startInput("screen")); document.querySelector("#stopInputButton").addEventListener("click", stopInput);
document.querySelector("#requestCoachButton").addEventListener("click", requestCoach); document.querySelector("#speakCoachButton").addEventListener("click", speakCoach);
document.querySelector("#lookupStudyButton").addEventListener("click", lookupOpenStudy);
elements.gameFormatInput.addEventListener("change", toggleTournamentFields);
document.querySelector("#saveFormatButton").addEventListener("click", saveGameFormat);

document.querySelector("#saveSeatsButton").addEventListener("click", async () => { const payload = collectSeatConfig(); if (!payload.seats.length) return showMessage("Add at least one named player before saving seats."); state.game = await api("/api/config/seats", { method: "POST", body: JSON.stringify(payload) }); await refreshPlayers(); renderSummary(); showMessage("Seat configuration saved."); });
document.querySelector("#saveObservedStatsButton").addEventListener("click", async () => {
  const player = elements.observedStatsPlayer.value;
  if (!player) return showMessage("Save seats first, then choose the player whose displayed stats you are storing.");
  const numberOrNull = (element) => element.value === "" ? null : Number(element.value);
  const payload = {
    hands_seen: numberOrNull(elements.observedHandsInput),
    vpip: numberOrNull(elements.observedVpipInput),
    pfr: numberOrNull(elements.observedPfrInput),
    aggression_factor: numberOrNull(elements.observedAfInput),
    source: "clubgg_profile",
    confidence: 1.0,
  };
  try {
    await api(`/api/players/${encodeURIComponent(player)}/observed-stats`, { method: "POST", body: JSON.stringify(payload) });
    await refreshPlayers();
    showMessage(`Stored the displayed ClubGG profile baseline for ${player}.`);
  } catch (error) { showMessage(`Could not save displayed stats: ${error.message}`); }
});
document.querySelector("#actionForm").addEventListener("submit", async (event) => { event.preventDefault(); state.game = await api("/api/action", { method: "POST", body: JSON.stringify({ player: elements.actionPlayer.value || null, action: elements.actionType.value, amount: elements.actionAmount.value ? Number(elements.actionAmount.value) : null, amount_type: elements.amountType.value, street: elements.streetInput.value }) }); elements.actionAmount.value = ""; renderSummary(); showMessage("Action logged."); });
document.querySelector("#speechForm").addEventListener("submit", async (event) => { event.preventDefault(); const text = elements.transcriptInput.value.trim(); if (!text) return; const result = await api("/api/speech/transcript", { method: "POST", body: JSON.stringify({ text }) }); state.game = result.state; elements.transcriptInput.value = ""; renderSummary(); showMessage(result.applied ? `Transcript applied: ${result.parsed.action}.` : `Parsed as ${result.parsed.action}; review it before logging.`); });
document.querySelector("#saveCardsButton").addEventListener("click", async () => {
  const cards = parseCardList(elements.heroCardsInput.value); const required = state.game?.game_variant === "plo5" ? 5 : 2;
  if (cards.length && cards.length !== required) return showMessage(`This game requires exactly ${required} hero cards.`);
  try {
    state.game = await api("/api/input/frame", { method: "POST", body: JSON.stringify({ source_id: "manual-context", observation: { hero_cards: cards, board_cards: parseCardList(elements.boardCardsInput.value), current_pot: elements.observedPotInput.value ? Number(elements.observedPotInput.value) : null } }) }).then((result) => result.state);
    renderSummary(); showMessage("Hand context saved.");
  } catch (error) { showMessage(`Could not save hand context: ${error.message}`); }
});
document.querySelector("#refreshVisionButton").addEventListener("click", async () => { const status = await api("/api/vision/status"); elements.visionStatusText.textContent = status.available ? `Local camera interface checked.${status.last_error ? ` ${status.last_error}` : ""}` : `Camera unavailable.${status.last_error ? ` ${status.last_error}` : ""}`; });
document.querySelector("#endHandButton").addEventListener("click", async () => { const winner = window.prompt("Winner name (optional)?") || null; const notes = window.prompt("Hand notes (optional)?") || null; const result = await api("/api/hand/end", { method: "POST", body: JSON.stringify({ winner, notes }) }); state.game = result.state; state.players = result.players; renderSummary(); renderProfiles(); showMessage("Hand archived and opponent memory updated."); });
document.querySelectorAll("[data-correction]").forEach((button) => button.addEventListener("click", async () => { const type = button.dataset.correction; const payload = { type }; if (type === "wrong_player") { payload.player = window.prompt("Correct player name?"); if (!payload.player) return; } if (type === "wrong_amount") { const amount = window.prompt("Correct amount?"); if (!amount) return; payload.amount = Number(amount); } if (type === "wrong_action") { payload.action = window.prompt("Correct action? raise, call, fold, check, bet, all_in"); if (!payload.action) return; } state.game = await api("/api/correction", { method: "POST", body: JSON.stringify(payload) }); renderSummary(); showMessage(type === "undo" ? "Last action removed." : "Last action corrected."); }));

function connectSocket() { const scheme = location.protocol === "https:" ? "wss" : "ws"; const socket = new WebSocket(`${scheme}://${location.host}/ws`); socket.addEventListener("open", () => { elements.socketStatus.textContent = "Live"; }); socket.addEventListener("message", (event) => { state.game = JSON.parse(event.data); renderSummary(); refreshPlayers().catch(() => showMessage("State updated, but profiles could not refresh.")); }); socket.addEventListener("close", () => { elements.socketStatus.textContent = "Reconnecting…"; window.setTimeout(connectSocket, 1500); }); }
async function loadInitialData() { await refreshState(); syncFormatForm(); await refreshPlayers(); if (!elements.seatRows.children.length) { createSeatRow({ seat: 1, player: "Hero", stack: 100, isHero: true }); createSeatRow({ seat: 2, player: "Alex", stack: 100, isHero: false }); } await refreshInputPreview(); showMessage("Ready. Choose game format, connect an input source, or enter the hand context manually."); }

loadInitialData().catch((error) => showMessage(`Failed to load app: ${error.message}`)); loadOpenStudyCatalog(); connectSocket(); window.setInterval(() => refreshInputPreview().catch(() => {}), 6000);
